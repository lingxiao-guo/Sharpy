#include <ATen/ops/nonzero.h>
#include <c10/core/DeviceType.h>
#include <reclib/depth_processing.h>
#include <reclib/optim/model_registration.h>
#include <reclib/tracking/sharpy_utils.h>

#include "reclib/cuda/device_info.cuh"
#include "reclib/dnn/dnn_utils.h"
#include "reclib/dnn/nvdiffrast_autograd.h"
#include "reclib/math/eigen_glm_interface.h"
#include "reclib/tracking/mano_rgbd_optim.h"
#include "reclib/tracking/sharpy.cuh"
#include "reclib/tracking/sharpy_tracker.h"

#if HAS_OPENCV_MODULE
#if HAS_DNN_MODULE
#if WITH_CUDA

#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <torch_tensorrt/core/compiler.h>
#include <torch_tensorrt/torch_tensorrt.h>

// Debug: save MANO state at each optimization step
int g_optim_debug_frame = 0;
static void save_optim_step_debug(
    const std::string& stage_name, int hand_idx,
    const torch::Tensor& mano_verts,
    const torch::Tensor& mano_joints,
    const torch::Tensor& faces,
    const torch::Tensor& pc_corrs,
    const torch::Tensor& mano_corrs,
    const torch::Tensor& extended_mask,
    const reclib::IntrinsicParameters& intr,
    const CpuMat& rgb,
    const reclib::Configuration& config) {

  if (g_optim_debug_frame >= 3) return;  // only debug first 3 frames

  fs::path debug_dir = fs::path(config["Pipeline"]["data_output_folder"].as<std::string>()) / "optim_debug";
  if (!fs::exists(debug_dir)) fs::create_directories(debug_dir);

  float fx = intr.focal_x_, fy = intr.focal_y_;
  float cx = intr.principal_x_, cy = intr.principal_y_;

  torch::Tensor verts_cpu = mano_verts.to(torch::kCPU).contiguous();
  torch::Tensor joints_cpu = mano_joints.to(torch::kCPU).contiguous();
  torch::Tensor faces_cpu = faces.to(torch::kCPU).contiguous();

  // Draw on RGB
  CpuMat overlay = rgb.clone();
  int n_verts = verts_cpu.sizes()[0];
  int n_faces = faces_cpu.sizes()[0];
  int n_joints = joints_cpu.sizes()[0];

  auto va = verts_cpu.accessor<float, 2>();
  auto fa = faces_cpu.accessor<int, 2>();

  // Project vertices
  std::vector<cv::Point2f> pts(n_verts);
  for (int i = 0; i < n_verts; i++) {
    float z = va[i][2];
    if (z > 0.001f) {
      pts[i] = cv::Point2f(fx * va[i][0] / z + cx, fy * va[i][1] / z + cy);
    } else {
      pts[i] = cv::Point2f(-1, -1);
    }
  }

  // Draw mesh
  cv::Scalar color = (hand_idx == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
  for (int f = 0; f < n_faces; f++) {
    int i0 = fa[f][0], i1 = fa[f][1], i2 = fa[f][2];
    if (pts[i0].x < 0 || pts[i1].x < 0 || pts[i2].x < 0) continue;
    cv::line(overlay, cv::Point(pts[i0]), cv::Point(pts[i1]), color, 1, cv::LINE_AA);
    cv::line(overlay, cv::Point(pts[i1]), cv::Point(pts[i2]), color, 1, cv::LINE_AA);
    cv::line(overlay, cv::Point(pts[i2]), cv::Point(pts[i0]), color, 1, cv::LINE_AA);
  }

  // Draw joints
  auto ja = joints_cpu.accessor<float, 2>();
  for (int j = 0; j < n_joints; j++) {
    float z = ja[j][2];
    if (z > 0.001f) {
      cv::circle(overlay, cv::Point(fx * ja[j][0] / z + cx, fy * ja[j][1] / z + cy), 5, cv::Scalar(0, 0, 255), -1);
    }
  }

  // Save overlay
  std::stringstream ss;
  ss << "f" << g_optim_debug_frame << "_hand" << hand_idx << "_" << stage_name << ".png";
  cv::imwrite((debug_dir / fs::path(ss.str())).string(), overlay);

  // Save correspondence pair stats
  if (pc_corrs.defined() && mano_corrs.defined() && pc_corrs.sizes()[0] > 0) {
    torch::Tensor pc_cpu = pc_corrs.to(torch::kCPU);
    torch::Tensor mc_cpu = mano_corrs.to(torch::kCPU);
    torch::Tensor dists = (pc_cpu - mc_cpu).norm(2, 1);
    std::cout << "[OPTIM STEP] " << stage_name << " hand" << hand_idx
              << " f" << g_optim_debug_frame
              << " | corr_dist: mean=" << dists.mean().item<float>()
              << " median=" << std::get<0>(dists.sort()).index({(int)(dists.sizes()[0]/2)}).item<float>()
              << " max=" << dists.max().item<float>()
              << " | verts_mean=(" << verts_cpu.index({torch::All, 0}).mean().item<float>()
              << "," << verts_cpu.index({torch::All, 1}).mean().item<float>()
              << "," << verts_cpu.index({torch::All, 2}).mean().item<float>() << ")"
              << std::endl;
  }
}

static std::vector<int> apose2joint = {1,  1,  2,  3,  4,  4,  5,  6,
                                       7,  7,  8,  9,  10, 10, 11, 12,
                                       13, 13, 13, 14, 14, 14, 15};

std::vector<bool> reclib::tracking::HandTracker::register_hand() {
  std::vector<bool> results(hand_states_.size());
  std::fill(results.begin(), results.end(), false);
  if (network_output_.size() == 0) {
    std::cout << "output is empty. quit." << std::endl;
    return results;
  }

  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    std::optional<bool> res = register_hand(i);
    if (res.has_value()) {
      results[i] = res.value();
      hand_states_[i].failed_ = !results[i];
    }
  }

  return results;
}

std::optional<bool> reclib::tracking::HandTracker::register_hand(
    unsigned int index) {
  torch::NoGradGuard guard;
  reclib::Configuration reg_config = config_.subconfig({"Registration"});
  bool update_meshes = !config_["Optimization"]["multithreading"].as<bool>();

  if (debug_) {
    timer_.look_and_reset();
  }

  HandState& state = hand_states_.at(index);

  if (state.corrs_state_.mano_corr_indices_.sizes()[0] == 0) {
    std::cout << "No correspondences. Abort." << std::endl;
    return false;
  }
  if (state.stage_ >= 1) {
    std::cout << "Hand is already registered." << std::endl;
    return {};
  }

  if (debug_) {
    timer_.look_and_reset();
  }

  // Ensure indices are on the same device as the tensors they index (PyTorch 2.6+ requirement)
  auto vm_dev = state.vertex_map_.device();
  torch::Tensor linearized_pc = state.vertex_map_.index(
      {state.nonzero_indices_.to(vm_dev).index({torch::All, 0}),
       state.nonzero_indices_.to(vm_dev).index({torch::All, 1}), torch::All});
  torch::Tensor pc_corrs =
      linearized_pc.index({state.corrs_state_.pc_corr_indices_linearized_.to(linearized_pc.device())});
  auto verts_dev = state.instance_->verts().device();
  torch::Tensor mano_corrs =
      state.instance_->verts().index({state.corrs_state_.mano_corr_indices_.to(verts_dev)});

  torch::Tensor incr_trans = torch::eye(4);

  // Debug: print registration input statistics
  {
    static int reg_debug_count = 0;
    if (reg_debug_count < 4) {
      torch::Tensor pc_corrs_cpu = pc_corrs.to(torch::kCPU);
      torch::Tensor mano_corrs_cpu = mano_corrs.to(torch::kCPU);
      torch::Tensor all_verts_cpu = state.instance_->verts().to(torch::kCPU);

      std::cout << "[REG DEBUG] Hand " << index << " (call #" << reg_debug_count << ")" << std::endl;
      std::cout << "  Num correspondences: " << pc_corrs_cpu.sizes()[0] << std::endl;
      std::cout << "  PC corrs (pointcloud): mean=("
                << pc_corrs_cpu.index({torch::All, 0}).mean().item<float>() << ", "
                << pc_corrs_cpu.index({torch::All, 1}).mean().item<float>() << ", "
                << pc_corrs_cpu.index({torch::All, 2}).mean().item<float>() << ")"
                << " range=[" << pc_corrs_cpu.min().item<float>() << ", " << pc_corrs_cpu.max().item<float>() << "]" << std::endl;
      std::cout << "  MANO corrs (vertices): mean=("
                << mano_corrs_cpu.index({torch::All, 0}).mean().item<float>() << ", "
                << mano_corrs_cpu.index({torch::All, 1}).mean().item<float>() << ", "
                << mano_corrs_cpu.index({torch::All, 2}).mean().item<float>() << ")"
                << " range=[" << mano_corrs_cpu.min().item<float>() << ", " << mano_corrs_cpu.max().item<float>() << "]" << std::endl;
      std::cout << "  ALL MANO verts: mean=("
                << all_verts_cpu.index({torch::All, 0}).mean().item<float>() << ", "
                << all_verts_cpu.index({torch::All, 1}).mean().item<float>() << ", "
                << all_verts_cpu.index({torch::All, 2}).mean().item<float>() << ")"
                << " range=[" << all_verts_cpu.min().item<float>() << ", " << all_verts_cpu.max().item<float>() << "]" << std::endl;

      // Distance between corresponding points
      torch::Tensor dists = (pc_corrs_cpu - mano_corrs_cpu).norm(2, 1);
      std::cout << "  Correspondence distances: mean=" << dists.mean().item<float>()
                << " median=" << std::get<0>(dists.sort()).index({(int)(dists.sizes()[0]/2)}).item<float>()
                << " min=" << dists.min().item<float>()
                << " max=" << dists.max().item<float>() << std::endl;
      reg_debug_count++;
    }
  }

  for (unsigned int i = 0; i < reg_config.ui("iterations"); i++) {
    if (reg_config.b("register_rotation")) {
      torch::Tensor rigid_trans =
          reclib::optim::pointToPointDirect(mano_corrs, pc_corrs);
      incr_trans = torch::matmul(rigid_trans, incr_trans);

      reclib::dnn::TorchVector aa_tensor = reclib::tracking::batch_mat2aa(
          incr_trans
              .index(
                  {torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)})
              .unsqueeze(0));

      state.instance_->set_trans(
          incr_trans.index({torch::indexing::Slice(0, 3), 3}));
      state.instance_->set_rot(aa_tensor.tensor());
      state.instance_->update(false, true, false, update_meshes);
    }

    // update tensor with updated vertices
    mano_corrs =
        state.instance_->verts().index({state.corrs_state_.mano_corr_indices_.to(state.instance_->verts().device())});

    if (reg_config.b("register_translation")) {
      torch::Tensor trans =
          reclib::optim::MeanTranslation(mano_corrs, pc_corrs);
      torch::Tensor T = torch::eye(4);
      T.index_put_({torch::indexing::Slice(0, 3), 3}, trans);
      incr_trans = torch::matmul(T, incr_trans);
      state.instance_->set_trans(
          incr_trans.index({torch::indexing::Slice(0, 3), 3}));
      state.instance_->update(false, true, false, update_meshes);
    }
  }

  // Debug: print post-registration state
  {
    static int post_reg_count = 0;
    if (post_reg_count < 4) {
      torch::Tensor post_verts = state.instance_->verts().to(torch::kCPU);
      torch::Tensor post_trans = state.instance_->trans().to(torch::kCPU);
      torch::Tensor post_rot = state.instance_->rot().to(torch::kCPU);

      std::cout << "[REG DEBUG POST] Hand " << index << std::endl;
      std::cout << "  Trans: " << post_trans << std::endl;
      std::cout << "  Rot: " << post_rot << std::endl;
      std::cout << "  Verts after reg: mean=("
                << post_verts.index({torch::All, 0}).mean().item<float>() << ", "
                << post_verts.index({torch::All, 1}).mean().item<float>() << ", "
                << post_verts.index({torch::All, 2}).mean().item<float>() << ")"
                << " range=[" << post_verts.min().item<float>() << ", " << post_verts.max().item<float>() << "]" << std::endl;

      // Recompute distances after registration
      torch::Tensor pc_corrs_cpu = linearized_pc.index(
          {state.corrs_state_.pc_corr_indices_linearized_.to(linearized_pc.device())}).to(torch::kCPU);
      torch::Tensor mano_corrs_post = post_verts.index(
          {state.corrs_state_.mano_corr_indices_.to(torch::kCPU)});
      torch::Tensor dists_post = (pc_corrs_cpu - mano_corrs_post).norm(2, 1);
      std::cout << "  Post-reg distances: mean=" << dists_post.mean().item<float>()
                << " median=" << std::get<0>(dists_post.sort()).index({(int)(dists_post.sizes()[0]/2)}).item<float>()
                << " min=" << dists_post.min().item<float>()
                << " max=" << dists_post.max().item<float>() << std::endl;
      std::cout << "  incr_trans:\n" << incr_trans << std::endl;
      post_reg_count++;
    }
  }

  if (debug_) {
    std::cout << "---- Registration: " << timer_.look_and_reset() << " ms."
              << std::endl;
  }

  if (config_["Debug"]["show_corr_lines"].as<bool>() &&
      config_["Optimization"]["multithreading"].as<bool>() == false) {
    visualize_correspondences(index);
  }
  state.stage_ = 1;
  return true;
}

std::vector<bool> reclib::tracking::HandTracker::optimize_hand(int stage) {
  std::vector<bool> results(hand_states_.size());
  std::fill(results.begin(), results.end(), false);
  if (network_output_.size() == 0) {
    std::cout << "output is empty. quit." << std::endl;
    return results;
  }

  for (unsigned int i = 0; i < hand_states_.size(); i++) {
    std::optional<bool> res = optimize_hand(i, stage);
    if (res.has_value()) {
      results[i] = res.value();
      hand_states_[i].failed_ = !results[i];
    }
  }

  return results;
}

std::optional<bool> reclib::tracking::HandTracker::optimize_hand(
    unsigned int index, int stage) {
  reclib::Configuration opt_config = config_.subconfig({"Optimization"});
  bool update_meshes = !config_["Optimization"]["multithreading"].as<bool>();

  HandState& state = hand_states_.at(index);

  // stage 0 is only executed once, directly after registration
  if (stage == 0 || !config_["Optimization"]["separate_stages"].as<bool>()) {
    if (state.stage_ >= 2) {
      return {};
    }
  }

  std::vector<bool> visibility_joints = state.corrs_state_.joints_visible_;

  // number of visible joint segments
  int num_visible = 0;
  if (debug_) {
    std::cout << "visibility: ";
    for (unsigned int i = 0; i < visibility_joints.size(); i++) {
      if (i > 0 && i - 1 % 3 == 0) {
        std::cout << std::endl;
        std::cout << "-----------" << std::endl;
      }
      if (visibility_joints[i] > 0) {
        num_visible++;
      }
      std::cout << (uint32_t)visibility_joints[i] << " ";
    }
    std::cout << std::endl;
  }

  // if a child within the finger segment is visible,
  // then all its parents are also visible
  // -> change their value to true
  for (unsigned int i = 0; i < (visibility_joints.size() - 1) / 3; i++) {
    for (unsigned int j = 2; j > 0; j--) {
      if (visibility_joints[i * 3 + j + 1]) {
        for (int k = j - 1; k >= 0; k--) {
          visibility_joints[i * 3 + k + 1] = true;
        }
        break;
      }
    }
  }

  float visible_percentage = num_visible / 16.f;

  if (state.corrs_state_.mano_corr_indices_.sizes()[0] == 0) {
    std::cout << "No correspondences. Abort." << std::endl;
    return false;
  }

  // upload to gpu
  if (!state.instance_->verts().is_cuda()) {
    if (state.instance_->model.hand_type == reclib::models::HandType::left) {
      mano_model_left_.gpu();
    } else {
      mano_model_right_.gpu();
    }
    state.instance_->gpu();
  }
  state.instance_->requires_grad(false);

  // load all parameters of current frame
  torch::Tensor trans;
  torch::Tensor shape;
  torch::Tensor rot;
  torch::Tensor pose;
  torch::Tensor apose;
  torch::Tensor pca;

  torch::Device dev = state.instance_->params.device();
  {
    torch::NoGradGuard guard;
    trans = state.instance_->trans().clone().detach();
    shape = state.instance_->shape().clone().detach();
    rot = state.instance_->rot().clone().detach();
    pca = state.instance_->hand_pca().clone().detach();
    apose = torch::matmul(state.instance_->model.hand_comps, pca.clone())
                .reshape({-1});
  }

  // load all parameters of previous frame
  torch::Tensor prev_trans;
  torch::Tensor prev_shape;
  torch::Tensor prev_rot;
  torch::Tensor prev_apose;
  torch::Tensor prev_pose;
  torch::Tensor prev_pca;
  torch::Tensor prev_visibility;

  if (sequence_frame_counter_ > 1) {
    torch::NoGradGuard guard;
    // load temporal information from last frame
    prev_trans = state.trans_.clone().detach().to(dev);
    prev_shape = state.shape_.clone().detach().to(dev);
    prev_rot = state.rot_.clone().detach().to(dev);
    prev_apose = state.pose_.clone().detach().to(dev);
  }

  torch::Tensor box = network_output_[2].index({(int)index}).contiguous();
  torch::Tensor mask = network_output_[3].index(
      {(int)index,
       torch::indexing::Slice(box.index({1}).item<int>(),
                              box.index({3}).item<int>()),
       torch::indexing::Slice(box.index({0}).item<int>(),
                              box.index({2}).item<int>())});
  torch::Tensor corr = network_output_[4].index({(int)index}).contiguous();
  auto vm_dev2 = state.vertex_map_.device();
  auto nzi = state.nonzero_indices_.to(vm_dev2);
  torch::Tensor linearized_vertex_map = state.vertex_map_.index(
      {nzi.index({torch::All, 0}),
       nzi.index({torch::All, 1}), torch::All});
  torch::Tensor linearized_normal_map = state.normal_map_.index(
      {nzi.index({torch::All, 0}),
       nzi.index({torch::All, 1}), torch::All});

  // crop the hand mask and insert it into a container of the
  // original image size -> everything is black except the hand region
  // within the box coordinates
  torch::Tensor extended_mask =
      torch::zeros({intrinsics_.image_height_, intrinsics_.image_width_},
                   torch::TensorOptions().device(dev));

  extended_mask.index_put_({torch::indexing::Slice(box.index({1}).item<int>(),
                                                   box.index({3}).item<int>()),
                            torch::indexing::Slice(box.index({0}).item<int>(),
                                                   box.index({2}).item<int>())},
                           mask);
  // Nvdiffrast uses OpenGL convention (y=0 at bottom).
  // Flip the mask and correspondence to match.
  extended_mask = torch::flip(extended_mask, 0).contiguous();

  corr = torch::flip(corr, 1).permute({1, 2, 0}).contiguous() *
         extended_mask.unsqueeze(-1);

  // compute view projection matrix as torch tensor
  mat4 proj = reclib::vision2graphics(intrinsics_.Matrix(), 0.01f, 1000.f,
                                      intrinsics_.image_width_,
                                      intrinsics_.image_height_);
  mat4 view = reclib::lookAt(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, -1, 0));
  mat4 vp = proj * view;
  torch::Tensor VP = reclib::dnn::eigen2torch<float, 4, 4>(vp, true).to(dev);

  // compute ground-truth correspondence image for Nvdiffrast
  torch::Tensor colors = mano_corr_space_.clone().to(dev);
  colors = colors.index({torch::None, "..."});

  // Debug: check correspondence space and rendered vs predicted corrs
  {
    static int corr_debug_count = 0;
    if (corr_debug_count < 2) {
      torch::Tensor colors_cpu = mano_corr_space_.to(torch::kCPU);
      torch::Tensor corr_cpu = corr.to(torch::kCPU);
      torch::Tensor mask_cpu = extended_mask.to(torch::kCPU);

      std::cout << "[CORR DEBUG] Hand " << index << std::endl;
      std::cout << "  mano_corr_space: shape=" << colors_cpu.sizes()
                << " min=" << colors_cpu.min().item<float>()
                << " max=" << colors_cpu.max().item<float>()
                << " mean=" << colors_cpu.mean().item<float>() << std::endl;
      for (int c = 0; c < 3; c++) {
        auto ch = colors_cpu.index({torch::All, c});
        std::cout << "    ch" << c << ": min=" << ch.min().item<float>()
                  << " max=" << ch.max().item<float>()
                  << " mean=" << ch.mean().item<float>() << std::endl;
      }

      // Network predicted corr (after mask and flip)
      torch::Tensor corr_masked = corr_cpu * mask_cpu.unsqueeze(-1);
      torch::Tensor corr_nonzero = corr_cpu.index({mask_cpu > 0});
      if (corr_nonzero.sizes()[0] > 0) {
        corr_nonzero = corr_nonzero.reshape({-1, 3});
        std::cout << "  Network corr (masked, non-zero): count=" << corr_nonzero.sizes()[0] << std::endl;
        for (int c = 0; c < 3; c++) {
          auto ch = corr_nonzero.index({torch::All, c});
          std::cout << "    ch" << c << ": min=" << ch.min().item<float>()
                    << " max=" << ch.max().item<float>()
                    << " mean=" << ch.mean().item<float>() << std::endl;
        }
      } else {
        std::cout << "  Network corr: NO non-zero pixels in mask!" << std::endl;
      }
      corr_debug_count++;
    }
  }

  // Debug: print actual VP matrix and mask position
  {
    static int vp_debug = 0;
    if (vp_debug < 2) {
      torch::Tensor vp_cpu = VP.to(torch::kCPU);
      std::cout << "[VP MATRIX] hand " << index << ":\n" << vp_cpu << std::endl;

      // Print mask stats
      torch::Tensor mask_cpu = extended_mask.to(torch::kCPU);
      auto mask_nonzero = torch::nonzero(mask_cpu);
      if (mask_nonzero.sizes()[0] > 0) {
        std::cout << "[MASK POS] rows: ["
                  << mask_nonzero.index({torch::All, 0}).min().item<int>() << ", "
                  << mask_nonzero.index({torch::All, 0}).max().item<int>() << "]"
                  << " cols: ["
                  << mask_nonzero.index({torch::All, 1}).min().item<int>() << ", "
                  << mask_nonzero.index({torch::All, 1}).max().item<int>() << "]"
                  << " total=" << mask_nonzero.sizes()[0] << " pixels" << std::endl;
      }
      vp_debug++;
    }
  }

  // Debug: check projection sanity
  {
    static int proj_debug_count = 0;
    if (proj_debug_count < 4) {
      torch::Tensor verts_cpu = state.instance_->verts().to(torch::kCPU);
      torch::Tensor vp_cpu = VP.to(torch::kCPU);
      // Project MANO vertices to screen space
      torch::Tensor verts_homo = torch::cat({verts_cpu, torch::ones({verts_cpu.sizes()[0], 1})}, 1);
      torch::Tensor verts_clip = torch::matmul(verts_homo, vp_cpu.transpose(0, 1));
      torch::Tensor verts_ndc = verts_clip.index({torch::All, torch::indexing::Slice(0, 2)}) /
                                verts_clip.index({torch::All, torch::indexing::Slice(3, 4)});
      // NDC to pixel: x_px = (ndc_x+1)/2 * W, y_px = (ndc_y+1)/2 * H
      float W = (float)intrinsics_.image_width_;
      float H = (float)intrinsics_.image_height_;
      torch::Tensor verts_px = torch::stack({
          (verts_ndc.index({torch::All, 0}) + 1.f) / 2.f * W,
          (verts_ndc.index({torch::All, 1}) + 1.f) / 2.f * H}, 1);

      std::cout << "[PROJ DEBUG] Hand " << index << " (stage " << stage << ")" << std::endl;
      std::cout << "  Image size: " << W << "x" << H << std::endl;
      std::cout << "  Verts 3D mean: (" << verts_cpu.index({torch::All, 0}).mean().item<float>()
                << ", " << verts_cpu.index({torch::All, 1}).mean().item<float>()
                << ", " << verts_cpu.index({torch::All, 2}).mean().item<float>() << ")" << std::endl;
      std::cout << "  Projected pixel mean: (" << verts_px.index({torch::All, 0}).mean().item<float>()
                << ", " << verts_px.index({torch::All, 1}).mean().item<float>() << ")" << std::endl;
      std::cout << "  Projected pixel range x: [" << verts_px.index({torch::All, 0}).min().item<float>()
                << ", " << verts_px.index({torch::All, 0}).max().item<float>() << "]" << std::endl;
      std::cout << "  Projected pixel range y: [" << verts_px.index({torch::All, 1}).min().item<float>()
                << ", " << verts_px.index({torch::All, 1}).max().item<float>() << "]" << std::endl;
      std::cout << "  Box: [" << box.index({0}).item<float>() << ", " << box.index({1}).item<float>()
                << ", " << box.index({2}).item<float>() << ", " << box.index({3}).item<float>() << "]" << std::endl;
      std::cout << "  Mask sum: " << extended_mask.sum().item<float>() << " pixels" << std::endl;
      proj_debug_count++;
    }
  }

  // Debug: save pre-optimization state
  if (stage == 0) {
    torch::NoGradGuard guard;
    auto t_pre = reclib::tracking::torch_lbs_pca_anatomic(
        state.instance_->model, trans, rot, shape, pca,
        reclib::tracking::compute_apose_matrix(state.instance_->model, shape), false);
    torch::Tensor pc_pre = linearized_vertex_map.index(
        {state.corrs_state_.pc_corr_indices_linearized_.to(linearized_vertex_map.device())}).to(torch::kCPU);
    torch::Tensor mc_pre = t_pre.first.index(
        {state.corrs_state_.mano_corr_indices_.to(t_pre.first.device())}).to(torch::kCPU);
    save_optim_step_debug("pre_optim", index, t_pre.first, t_pre.second,
        state.instance_->model.faces, pc_pre, mc_pre,
        extended_mask, intrinsics_, rgb_, config_);
  }

  // initialize Nvdiffrast
  int device = reclib::getDevice();
  RasterizeCRStateWrapper context(device);

  // pre-compute the cross_matrix, which transforms the original MANO pose
  // orientation to anatomically correct axes
  torch::Tensor cross_matrix =
      reclib::tracking::compute_apose_matrix(state.instance_->model, shape);

  std::vector<torch::Tensor> params;
  params.push_back(rot);
  params.push_back(pca);
  if (visible_percentage > 0.5) {
    params.push_back(shape);
  }

  // Configure Adam Options
  torch::optim::AdamOptions adam_opt;
  adam_opt.amsgrad() = true;
  if (sequence_frame_counter_ <= 1) {
    adam_opt.set_lr(opt_config["adam_lr_initialization"].as<float>());
  } else {
    adam_opt.set_lr(opt_config["adam_lr"].as<float>());
  }
  adam_opt.weight_decay() = opt_config["adam_weight_decay"].as<float>();

  // Configure LBFGS Options
  torch::optim::LBFGSOptions lbfgs_opt;
  if (sequence_frame_counter_ <= 1) {
    lbfgs_opt.max_iter(
        opt_config["lbfgs_inner_iterations_initialization"].as<int>());
  } else {
    lbfgs_opt.max_iter(opt_config["lbfgs_inner_iterations"].as<int>());
  }
  lbfgs_opt.line_search_fn() = "strong_wolfe";

  if (debug_)
    std::cout << "---- Preparation: " << timer_.look_and_reset() << " ms."
              << std::endl;

  for (unsigned int i = 0;
       i < opt_config["outer_iterations"].as<unsigned int>(); i++) {
    torch::optim::LBFGS optim_rot_pose_trans =
        torch::optim::LBFGS(params, lbfgs_opt);

    auto loss_func_rot_apose = [&]() {
      // ---------------------------------------------
      // Compute LBS
      // ---------------------------------------------
      std::pair<torch::Tensor, torch::Tensor> t;
      if (pca.requires_grad()) {
        // optimize over PCA
        t = reclib::tracking::torch_lbs_pca_anatomic(state.instance_->model,
                                                     trans, rot, shape, pca,
                                                     cross_matrix, false);
      } else {
        // optimize over pose
        t = reclib::tracking::torch_lbs_anatomic(state.instance_->model, trans,
                                                 rot, shape, apose,
                                                 cross_matrix, false);
      }

      // ---------------------------------------------
      // Silhouette Term
      // ---------------------------------------------
      const auto [silhouette_term, iou_pred] = rasterizer_loss(
          state, opt_config, context, t.first, colors, VP, corr, extended_mask);

      // ---------------------------------------------
      // Data Term
      // ---------------------------------------------
      torch::Tensor data_term =
          data_loss(state, opt_config, t.first, linearized_vertex_map,
                    linearized_normal_map, iou_pred);

      // ---------------------------------------------
      // Regularization terms
      // ---------------------------------------------
      std::vector<torch::Tensor> reg_losses =
          param_regularizer(opt_config, trans, rot, pose, pca, shape);
      std::vector<torch::Tensor> temp_losses = temp_regularizer(
          opt_config, {trans, prev_trans}, {rot, prev_rot}, {pose, prev_pose},
          {pca, prev_pca}, {shape, prev_shape}, iou_pred, data_term);

      // ---------------------------------------------
      // Compute final loss
      // ---------------------------------------------

      torch::Tensor loss = data_term + silhouette_term;
      for (unsigned int i = 0; i < reg_losses.size(); i++) {
        loss = loss + reg_losses[i];
      }
      for (unsigned int i = 0; i < temp_losses.size(); i++) {
        loss = loss + temp_losses[i];
      }

      loss.backward();

      if (1 && apose.requires_grad()) {
        for (int i = 0; i < apose.sizes()[0]; i++) {
          int joint = apose2joint[i];
          // do not update gradients of pose parameters in which
          // the corresponding joints were not visible
          if (!(bool)visibility_joints[joint]) {
            apose.mutable_grad().index({i}) = 0;
          }
        }
      }

      if (debug_)
        std::cout << "--------------------------------------" << std::endl;

      return loss;
    };

    // LBFGS requires a wrapping function
    auto lbfgs_loss_func_wrapper = [&]() {
      optim_rot_pose_trans.zero_grad();
      torch::Tensor loss = loss_func_rot_apose();
      return loss;
    };

    // -----------------------------------------------------------------------------------------------------------
    // Optimization Stage 0
    // -----------------------------------------------------------------------------------------------------------
    if (stage == 0 || !config_["Optimization"]["separate_stages"].as<bool>()) {
      pca.set_requires_grad(true);
      rot.set_requires_grad(true);
      shape.set_requires_grad(true);

      {
        torch::Tensor prev_apose_optim = apose.clone().detach();
        torch::Tensor prev_pca_optim = pca.clone().detach();

        optim_rot_pose_trans.step(lbfgs_loss_func_wrapper);

        if (debug_)
          std::cout << "diff: "
                    << (pca - prev_pca_optim).abs().sum().item<float>()
                    << std::endl;
      }

      if (debug_)
        std::cout << "[ITERATIONS]:" << lbfgs_opt.max_iter() << std::endl;

      pca.set_requires_grad(false);
      rot.set_requires_grad(false);
      shape.set_requires_grad(false);

      // Debug: save MANO state after Stage 0 (L-BFGS initialization)
      {
        torch::NoGradGuard guard;
        auto t0 = reclib::tracking::torch_lbs_pca_anatomic(
            state.instance_->model, trans, rot, shape, pca, cross_matrix, false);
        torch::Tensor pc_corrs_s0 = linearized_vertex_map.index(
            {state.corrs_state_.pc_corr_indices_linearized_.to(linearized_vertex_map.device())}).to(torch::kCPU);
        torch::Tensor mano_corrs_s0 = t0.first.index(
            {state.corrs_state_.mano_corr_indices_.to(t0.first.device())}).to(torch::kCPU);
        save_optim_step_debug("stage0_lbfgs", index, t0.first, t0.second,
            state.instance_->model.faces, pc_corrs_s0, mano_corrs_s0,
            extended_mask, intrinsics_, rgb_, config_);
      }

      state.stage_ = 2;
    }

    if (!config_["Optimization"]["separate_stages"].as<bool>()) {
      torch::NoGradGuard guard;
      // recompute apose from optimized pca
      apose = torch::matmul(state.instance_->model.hand_comps, pca.clone())
                  .reshape({-1});
    }

    // -----------------------------------------------------------------------------------------------------------
    // Optimization Stage 1
    // -------------------------------------------------------------------------------------------------------

    torch::optim::Adam optim_rot_pose_trans_refined =
        torch::optim::Adam({trans, rot, shape, apose}, adam_opt);
    torch::optim::Adam optim_rot_pose_refined_ =
        torch::optim::Adam({shape, apose}, adam_opt);
    torch::optim::Adam optim_trans_rot =
        torch::optim::Adam({trans, rot}, adam_opt);

    int epochs = opt_config.i("adam_lr_epochs");
    if (sequence_frame_counter_ <= 1) {
      epochs = opt_config.i("adam_lr_epochs_initial");
    }

    torch::optim::StepLR optim_rot_pose_refined = torch::optim::StepLR(
        optim_rot_pose_refined_, epochs, opt_config.f("adam_lr_step_size"));
    torch::optim::StepLR optim_trans = torch::optim::StepLR(
        optim_trans_rot, epochs, opt_config.f("adam_lr_step_size"));

    if (stage == 1 || !config_["Optimization"]["separate_stages"].as<bool>()) {
      int termination_iter = opt_config["adam_inner_iterations"].as<int>();
      if (sequence_frame_counter_ <= 1 || state.loss_ < 0) {
        termination_iter =
            opt_config["adam_inner_iterations_initialization"].as<int>();
      }

      apose.set_requires_grad(true);
      rot.set_requires_grad(true);
      shape.set_requires_grad(true);
      trans.set_requires_grad(true);

      torch::Tensor best_apose = apose.clone().detach();
      torch::Tensor best_trans = trans.clone().detach();
      torch::Tensor best_rot = rot.clone().detach();
      torch::Tensor best_shape = shape.clone().detach();
      float best_loss = -1;

      int iter = 0;
      torch::Tensor prev_apose_optim;
      torch::Tensor prev_rot_optim;
      torch::Tensor prev_trans_optim;
      torch::Tensor loss;

      while (true) {
        prev_apose_optim = apose.clone().detach();
        prev_rot_optim = rot.clone().detach();
        prev_trans_optim = trans.clone().detach();

        if (opt_config.b("use_flip_flop")) {
          apose.set_requires_grad(true);
          rot.set_requires_grad(false);
          shape.set_requires_grad(true);
          trans.set_requires_grad(false);

          // optimize only apose + shape
          optim_rot_pose_refined_.zero_grad();
          // loss_func_rot_apose takes by far the most time, split between this
          // call and the one below
          loss = loss_func_rot_apose();
          optim_rot_pose_refined_.step();
          optim_rot_pose_refined.step();

          if (apose.isnan().any().item<bool>()) {
            throw std::runtime_error("Apose is NaN");
          }
          if (shape.isnan().any().item<bool>()) {
            throw std::runtime_error("Shape is NaN");
          }

          // optimize only rot + trans
          apose.set_requires_grad(false);
          rot.set_requires_grad(true);
          shape.set_requires_grad(false);
          trans.set_requires_grad(true);

          optim_trans_rot.zero_grad();
          loss = loss_func_rot_apose();
          optim_trans_rot.step();
          optim_trans.step();

          if (rot.isnan().any().item<bool>()) {
            throw std::runtime_error("Rot is NaN");
          }
          if (trans.isnan().any().item<bool>()) {
            throw std::runtime_error("Trans is NaN");
          }

        } else {
          // optimize only rot + trans + pose + shape
          optim_rot_pose_trans_refined.zero_grad();
          loss = loss_func_rot_apose();
          optim_rot_pose_trans_refined.step();
        }

        // store the best parameters of the whole optimization
        if (best_loss == -1) {
          best_apose = apose.clone().detach();
          best_trans = trans.clone().detach();
          best_rot = rot.clone().detach();
          best_shape = shape.clone().detach();
          best_loss = loss.clone().detach().item<float>();
        } else if (iter > 0 && (best_loss > loss).all().item<bool>()) {
          best_apose = apose.clone().detach();
          best_trans = trans.clone().detach();
          best_rot = rot.clone().detach();
          best_shape = shape.clone().detach();
          best_loss = loss.clone().detach().item<float>();
        }

        // compute the difference in parameter updates
        float diff = (apose - prev_apose_optim).abs().sum().item<float>();
        diff += (rot - prev_rot_optim).abs().sum().item<float>();
        diff += (trans - prev_trans_optim).abs().sum().item<float>();

        // terminate algorithm if parameters barely changed
        if ((diff <= opt_config["adam_termination_eps"].as<float>()) ||
            iter > termination_iter) {
          break;
        }

        iter++;
      }
      if (debug_) {
        std::cout << "Loss: " << loss << std::endl;
        std::cout << "Best loss: " << best_loss << std::endl;
        std::cout << "Last frame best loss: " << state.loss_ << std::endl;
        std::cout << "Difference in losses: "
                  << (best_loss - state.loss_) / state.loss_ << std::endl;
      }

      apose.set_requires_grad(false);
      rot.set_requires_grad(false);
      shape.set_requires_grad(false);
      trans.set_requires_grad(false);

      // Debug: save MANO state after Stage 1 (Adam refinement)
      {
        torch::NoGradGuard guard;
        auto t1 = reclib::tracking::torch_lbs_anatomic(
            state.instance_->model, best_trans, best_rot, best_shape, best_apose,
            cross_matrix, false);
        torch::Tensor pc_corrs_s1 = linearized_vertex_map.index(
            {state.corrs_state_.pc_corr_indices_linearized_.to(linearized_vertex_map.device())}).to(torch::kCPU);
        torch::Tensor mano_corrs_s1 = t1.first.index(
            {state.corrs_state_.mano_corr_indices_.to(t1.first.device())}).to(torch::kCPU);
        save_optim_step_debug("stage1_adam", index, t1.first, t1.second,
            state.instance_->model.faces, pc_corrs_s1, mano_corrs_s1,
            extended_mask, intrinsics_, rgb_, config_);

        // Also print the pose parameters to check if they changed
        std::cout << "[OPTIM PARAMS] hand" << index << " f" << g_optim_debug_frame
                  << " | best_apose norm=" << best_apose.norm().item<float>()
                  << " max=" << best_apose.abs().max().item<float>()
                  << " | pca norm=" << pca.norm().item<float>()
                  << " | rot=" << best_rot.to(torch::kCPU)
                  << " | trans=" << best_trans.to(torch::kCPU)
                  << std::endl;
      }

      // Restore best parameters found during optimization
      {
        torch::NoGradGuard guard;
        apose = best_apose.clone();
        trans = best_trans.clone();
        rot = best_rot.clone();
        shape = best_shape.clone();
      }

      // compute PCA since pose updates are stored as a PCA within the instance
      pca = torch::matmul(state.instance_->model.hand_comps.inverse(),
                          apose.unsqueeze(1))
                .reshape({-1});
      if (pca.isnan().any().item<bool>()) {
        throw std::runtime_error("PCA is NaN");
      }

      if (debug_) {
        std::cout << "[ITERATIONS]:" << iter << ", " << termination_iter
                  << std::endl;
      }

      // Reset the tracking state if the relative or absolute loss
      // becomes too high
      if (state.loss_ > 0 && (best_loss - state.loss_) / state.loss_ >
                                 opt_config.f("loss_relative_threshold")) {
        if (debug_) {
          std::cout << "Loss higher than last one: " << best_loss << " <-> "
                    << state.loss_ << std::endl;
        }

        return false;
      }
      if (state.loss_ > 0 &&
          best_loss > opt_config.f("loss_absolute_threshold")) {
        if (debug_) {
          std::cout << "Loss higher than absolute thresh: " << best_loss
                    << " <-> " << opt_config.f("loss_absolute_threshold")
                    << std::endl;
        }

        return false;
      }
      state.loss_ = best_loss;
      state.stage_ = 3;
    }

    // -----------------------------------------------------------------------------------------------------------
    // Optimization Stage 2
    // -------------------------------------------------------------------------------------------------------

    if (stage == 2) {
      shape.set_requires_grad(true);
      apose.set_requires_grad(false);
      trans.set_requires_grad(false);
      rot.set_requires_grad(false);
      pca.set_requires_grad(false);

      // only optimize the hand shape
      torch::optim::Adam optim_shape = torch::optim::Adam({shape}, adam_opt);

      int iter = 0;
      while (true) {
        torch::Tensor prev_shape_optim = shape.clone().detach();

        optim_shape.zero_grad();
        torch::Tensor loss = loss_func_rot_apose();
        optim_shape.step();

        if (debug_)
          std::cout << "diff: "
                    << (shape - prev_shape_optim).abs().sum().item<float>()
                    << std::endl;

        if ((shape - prev_shape_optim).abs().sum().item<float>() <=
                opt_config["adam_termination_eps"].as<float>() ||
            iter > opt_config["adam_inner_iterations"].as<int>()) {
          break;
        }

        iter++;
      }
      shape.set_requires_grad(false);

      if (debug_) std::cout << "Iterations: " << iter << std::endl;
      state.stage_ = 4;
    }
  }

  if (debug_)
    std::cout << "---- Optimization: " << timer_.look_and_reset() << " ms."
              << std::endl;

  int pca_len = 45;
  if (state.instance_->use_anatomic_pca_) {
    pca_len = 23;
  }

  state.instance_->set_trans(trans);
  state.instance_->params.index_put_(
      {torch::indexing::Slice(3 + 3 + 45, 3 + 3 + 45 + pca_len)}, pca);
  state.instance_->params.index_put_(
      {torch::indexing::Slice(3 + 3 + 45 + pca_len, 3 + 3 + 45 + pca_len + 10)},
      shape);
  state.instance_->set_rot(rot);
  state.instance_->update(false, true, false, update_meshes);

  if (debug_)
    std::cout << "---- Postprocessing: " << timer_.look_and_reset() << " ms."
              << std::endl;

  if (config_["Debug"]["show_corr_lines"].as<bool>() &&
      config_["Optimization"]["multithreading"].as<bool>() == false) {
    visualize_correspondences(index);
  }
  return true;
}

torch::Tensor reclib::tracking::HandTracker::data_loss(
    HandState& state, reclib::Configuration& opt_config,
    torch::Tensor mano_verts, torch::Tensor linearized_vertex_map,
    torch::Tensor linearized_normal_map, torch::Tensor iou_pred) {
  // iou is a weighting factor between data term and silhouette term
  // the greater iou -> higher weight for silhouette
  // the smaller iou -> higher weight for data (because less overlap)
  float weight = std::exp(-iou_pred.item<float>() + 1) *
                 opt_config["data_weight"].as<float>();

  torch::Tensor point2point =
      weight * (linearized_vertex_map.index(
                    {state.corrs_state_.pc_corr_indices_linearized_}) -
                mano_verts.index({state.corrs_state_.mano_corr_indices_}));

  torch::Tensor point2plane =
      weight *
      torch::bmm(point2point.index({torch::All, torch::None, torch::All}),
                 linearized_normal_map
                     .index({state.corrs_state_.pc_corr_indices_linearized_})
                     .index({torch::All, torch::All, torch::None}));

  torch::Tensor loss = weight * (point2point.abs().mean() * 0.33 +
                                 point2plane.abs().mean() * 0.66);

  if (debug_) {
    std::cout << "-- data: " << loss.item<float>() << ", weight: " << weight
              << " (iou:" << iou_pred.item<float>() << ")" << std::endl;
  }

  return loss;
}

std::pair<torch::Tensor, torch::Tensor>
reclib::tracking::HandTracker::rasterizer_loss(
    HandState& state, reclib::Configuration& opt_config,
    RasterizeCRStateWrapper& context, torch::Tensor mano_verts,
    torch::Tensor colors, torch::Tensor VP, torch::Tensor corr,
    torch::Tensor extended_mask) {
  torch::Device dev = mano_verts.device();
  int w = intrinsics_.image_width_;
  int h = intrinsics_.image_height_;
  std::tuple<int, int> res = std::make_tuple(h, w);

  torch::Tensor verts = mano_verts.contiguous();

  torch::Tensor ones =
      torch::ones({verts.sizes()[0], 1}, torch::TensorOptions().device(dev))
          .contiguous();
  // compute vertices in homogeneous coordinates
  torch::Tensor verts_hom = torch::cat({verts, ones}, 1).contiguous();
  // transform to clip space
  torch::Tensor verts_clip =
      torch::matmul(verts_hom, VP.transpose(1, 0)).contiguous();
  verts_clip = verts_clip.index({torch::None, "..."});
  torch::Tensor pos_idx = state.instance_->model.faces;

  // ---------------------------------------------
  // Apply Nvdiffrast (differential Rasterizer)
  // ---------------------------------------------
  std::vector<torch::Tensor> rast_out =
      reclib::dnn::rasterize(context, verts_clip, pos_idx, res);
  std::vector<torch::Tensor> interp_out =
      reclib::dnn::interpolate(colors, rast_out[0], pos_idx);
  std::vector<torch::Tensor> antialias_out =
      reclib::dnn::antialias(interp_out[0], rast_out[0], verts_clip, pos_idx);
  // predicted MANO colors (aka canonical coordinates) from Nvdiffrast
  torch::Tensor color_pred = antialias_out[0].squeeze(0);

  // Debug: save rendered MANO vs mask overlay
  {
    static int rast_debug_count = 0;
    if (rast_debug_count < 4) {
      fs::path debug_dir = config_["Pipeline"]["data_output_folder"].as<std::string>();
      fs::path rast_dir = debug_dir / "rasterizer_debug";
      if (!fs::exists(rast_dir)) fs::create_directories(rast_dir);

      // Rendered MANO silhouette (binary)
      torch::Tensor rendered_sil = (color_pred.sum(2) > 0).to(torch::kFloat32).to(torch::kCPU);
      // Network mask (already y-flipped)
      torch::Tensor mask_vis = extended_mask.to(torch::kFloat32).to(torch::kCPU);

      // Debug: print exact pixel positions of mask and rendered MANO
      auto mask_nz = torch::nonzero(mask_vis);
      auto rend_nz = torch::nonzero(rendered_sil);
      if (mask_nz.sizes()[0] > 0) {
        std::cout << "[OVERLAY] Mask rows: [" << mask_nz.index({torch::All, 0}).min().item<int>()
                  << ", " << mask_nz.index({torch::All, 0}).max().item<int>() << "]" << std::endl;
      }
      if (rend_nz.sizes()[0] > 0) {
        std::cout << "[OVERLAY] Rendered rows: [" << rend_nz.index({torch::All, 0}).min().item<int>()
                  << ", " << rend_nz.index({torch::All, 0}).max().item<int>() << "]" << std::endl;
      }

      // Save as 3-channel overlay: R=mask, G=rendered, B=0
      torch::Tensor overlay = torch::zeros({h, w, 3});
      overlay.index({torch::All, torch::All, 0}) = mask_vis;      // red = network mask
      overlay.index({torch::All, torch::All, 1}) = rendered_sil;  // green = rendered MANO
      CpuMat overlay_cv = reclib::dnn::torch2cv(overlay);
      overlay_cv.convertTo(overlay_cv, CV_8UC3, 255.0);

      std::stringstream ss;
      ss << "overlay_hand" << (int)state.instance_->model.hand_type << "_" << rast_debug_count << ".png";
      cv::imwrite((rast_dir / fs::path(ss.str())).string(), overlay_cv);

      // Also save the rendered correspondence image
      torch::Tensor color_vis = color_pred.to(torch::kCPU).contiguous();
      CpuMat color_cv = reclib::dnn::torch2cv(color_vis);
      color_cv.convertTo(color_cv, CV_8UC3, 255.0);
      std::stringstream ss2;
      ss2 << "rendered_corr_hand" << (int)state.instance_->model.hand_type << "_" << rast_debug_count << ".png";
      cv::imwrite((rast_dir / fs::path(ss2.str())).string(), color_cv);

      rast_debug_count++;
    }
  }

  // compute nonzero pixels within prediction
  torch::Tensor positive_samples_pred = (color_pred.sum(2) > 0).sum();
  // compute pixel intersection between prediction and ground truth
  torch::Tensor iou_intersection =
      torch::logical_and(color_pred.sum(2) > 0, extended_mask);
  // compute pxiel union between prediction and ground truth
  torch::Tensor iou_union =
      torch::logical_or(color_pred.sum(2) > 0, extended_mask);
  // compute intersection over union (iou)
  torch::Tensor iou_pred = iou_intersection.sum() / iou_union.sum();

  torch::Tensor loss = opt_config["silhouette_weight"].as<float>() *
                       torch::l1_loss(corr, color_pred, torch::Reduction::Sum) /
                       extended_mask.sum();

  if (debug_) {
    std::cout << "-- silhouette: " << loss.item<float>() << std::endl;
  }

  return std::make_pair(loss, iou_pred);
}

std::vector<torch::Tensor> reclib::tracking::HandTracker::param_regularizer(
    reclib::Configuration& opt_config, torch::Tensor trans, torch::Tensor rot,
    torch::Tensor pose, torch::Tensor pca, torch::Tensor shape) {
  std::vector<torch::Tensor> losses;

  if (pose.requires_grad()) {
    losses.push_back(opt_config["param_regularizer_pose_weight"].as<float>() *
                     pose.norm() * pose.norm());

    if (debug_) {
      std::cout << "-- param reg (pose): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  if (shape.requires_grad()) {
    losses.push_back(opt_config["param_regularizer_shape_weight"].as<float>() *
                     shape.norm() * shape.norm());

    if (debug_) {
      std::cout << "-- param reg (shape): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }

  return losses;
}

std::vector<torch::Tensor> reclib::tracking::HandTracker::temp_regularizer(
    reclib::Configuration& opt_config,
    std::pair<torch::Tensor, torch::Tensor> trans,
    std::pair<torch::Tensor, torch::Tensor> rot,
    std::pair<torch::Tensor, torch::Tensor> pose,
    std::pair<torch::Tensor, torch::Tensor> pca,
    std::pair<torch::Tensor, torch::Tensor> shape, torch::Tensor iou_pred,
    torch::Tensor data_term) {
  std::vector<torch::Tensor> losses;
  if (sequence_frame_counter_ <= 1) return losses;

  float regularization_weight_iou = std::exp(-iou_pred.item<float>() + 1);
  float regularization_weight_data = std::log1p(data_term.item<float>());

  if (debug_) {
    std::cout << "[TEMP REG] iou_w: " << regularization_weight_iou
              << " data_w: " << regularization_weight_data << std::endl;
  }

  // ----------- APOSE -----------
  if (pose.first.requires_grad() && pose.second.sizes()[0] > 0) {
    losses.push_back(
        opt_config["temp_regularizer_pose_weight"].as<float>() *
        (regularization_weight_iou + regularization_weight_data) *
        (torch::mse_loss(pose.first, pose.second, torch::Reduction::None) *
         (pose.second > 0))
            .sum());

    if (debug_) {
      std::cout << "-- temp reg (pose): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  if (shape.first.requires_grad() && shape.second.sizes()[0] > 0) {
    losses.push_back(
        opt_config["temp_regularizer_shape_weight"].as<float>() *
        torch::nn::functional::mse_loss(shape.first, shape.second));

    if (debug_) {
      std::cout << "-- temp reg (shape): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  if (trans.first.requires_grad() && trans.second.sizes()[0] > 0) {
    losses.push_back(
        opt_config["temp_regularizer_trans_weight"].as<float>() *
        (regularization_weight_iou + regularization_weight_data) *
        torch::nn::functional::mse_loss(trans.first, trans.second));

    if (debug_) {
      std::cout << "-- temp reg (trans): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  if (rot.first.requires_grad() && rot.second.sizes()[0] > 0) {
    losses.push_back(opt_config["temp_regularizer_rot_weight"].as<float>() *
                     (regularization_weight_iou + regularization_weight_data) *
                     torch::nn::functional::mse_loss(rot.first, rot.second));

    if (debug_) {
      std::cout << "-- temp reg (rot): "
                << losses[losses.size() - 1].item<float>() << std::endl;
    }
  }
  return losses;
}

#endif  // WITH_CUDA
#endif  // HAS_DNN_MODULE
#endif  // HAS_OPENCV_MODULE