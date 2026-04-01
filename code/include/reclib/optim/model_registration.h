#if __unix__

#ifndef RECLIB_OPTIM_MODEL_REGISTRATION_H
#define RECLIB_OPTIM_MODEL_REGISTRATION_H

#include "reclib/data_types.h"
#include "reclib/fusion/tsdf_volume.h"
#include "reclib/models/model_config.h"
#include "reclib/models/smpl.h"
#include "reclib/optim/correspondences.h"
#include "reclib/optim/registration.h"
#include "sophus/se3.hpp"

namespace reclib {
namespace optim {

#if HAS_OPENCV_MODULE
// Special registration from mano hand to 3D point cloud
// with nearest neighbor correspondences
// and direct p2p optimization
mat4 registerMANO2Pointcloud(const reclib::Configuration& config,
                             const float* mano_vertices,
                             uint32_t mano_n_vertices, const float* pc_vertices,
                             uint32_t pc_n_vertices,
                             const float* mano_normals = nullptr,
                             const float* pc_normals = nullptr,
                             const Eigen::Matrix<float, 4, 4>& mano_pre_trans =
                                 Eigen::Matrix<float, 4, 4>::Identity(),
                             const Eigen::Matrix<float, 4, 4>& pc_pre_trans =
                                 Eigen::Matrix<float, 4, 4>::Identity(),
                             vec3* pca_mean = nullptr);

// More generalized registration from mano hand to 2D vertex map
// uses direct p2p optimization
Sophus::SE3<float> registerMANO2Pointcloud(
    reclib::Configuration config,
    const reclib::models::ModelInstance<reclib::models::MANOConfig>& mano,
    const float* pc_vertices, const reclib::ExtrinsicParameters& cam_extr,
    const reclib::IntrinsicParameters& cam_intr,
    const float* mano_normals = nullptr, const float* pc_normals = nullptr,
    const Eigen::Matrix<float, 4, 4>& mano_pre_trans =
        Eigen::Matrix<float, 4, 4>::Identity(),
    const Eigen::Matrix<float, 4, 4>& pc_pre_trans =
        Eigen::Matrix<float, 4, 4>::Identity(),
    vec3* pca_mean = nullptr, vec3* mano_mean = nullptr);
#endif  // HAS_OPENCV_MODULE

// SMPL2SDFFunctor commented out: references removed LBS function
// #if HAS_OPENCV_MODULE
// template <class ModelConfig>
// struct SMPL2SDFFunctor { ... };

#if HAS_OPENCV_MODULE

// Localizes the mean of the hand shape
// Underlying assumption: the point cloud is an arm
// When computing the mean of an arm, it is typically biased by the length
// of the arm itself
// For Hand registration we need the mean of the hand itself
vec3 PCAHandMean(const float* pc_vertices, uint32_t pc_n_vertices,
                 float variance_thresh = 0.8);

#endif  // HAS_OPENCV_MODULE

}  // namespace optim
}  // namespace reclib

#endif

#endif  //__unix__