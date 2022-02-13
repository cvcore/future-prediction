#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
/*
Based on the DSAC++ and ESAC code.
https://github.com/vislearn/LessMore
https://github.com/vislearn/esac

Copyright (c) 2016, TU Dresden
Copyright (c) 2020, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <torch/extension.h>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "thread_rand.h"
#include "stop_watch.h"

#include "dsacstar_types.h"
#include "dsacstar_util.h"
#include "dsacstar_util_rgbd.h"
#include "dsacstar_loss.h"
#include "dsacstar_derivative.h"

#define MAX_REF_STEPS 50 // max pose refienment iterations
#define MAX_HYPOTHESES_TRIES 10000 // repeat sampling x times hypothesis if hypothesis is invalid

/**
 * @brief Estimate a camera pose based on a scene coordinate prediction
 * @param sceneCoordinatesSrc Scene coordinate prediction, (1x3xHxW) with 1=batch dimension (only batch_size=1 supported atm), 3=scene coordainte dimensions, H=height and W=width.
 * @param outPoseSrc Camera pose (output parameter), (4x4) tensor containing the homogeneous camera tranformation matrix.
 * @param ransacHypotheses Number of RANSAC iterations.
 * @param inlierThreshold Inlier threshold for RANSAC in px.
 * @param focalLength Focal length of the camera in px.
 * @param ppointX Coordinate (X) of the prinicpal points.
 * @param ppointY Coordinate (Y) of the prinicpal points.
 * @param inlierAlpha Alpha parameter for soft inlier counting.
 * @param maxReprojError Reprojection errors are clamped above this value (px).
 * @param subSampling Sub-sampling  of the scene coordinate prediction wrt the input image.
 */
std::vector<torch::Tensor> dsacstar_forward(
	const torch::Tensor coords2dSrc,
	const torch::Tensor coords3dSrc,
	const torch::Tensor cameraMatrixSrc,
	int ransacHypotheses, 
	float inlierThreshold,
	float inlierAlpha,
	float maxReprojError,
	int minPoints,
	int randomSeed)
{
	ThreadRand::init(randomSeed);

	// Access to tensor objects
	dsacstar::coord_t coords2d = coords2dSrc.accessor<float, 3>();
    dsacstar::coord_t coords3d = coords3dSrc.accessor<float, 3>();
    dsacstar::coord_t cameraMatrix = cameraMatrixSrc.accessor<float, 3>();

    int B = coords2d.size(0);
    int MAX_PT_COUNT = coords2d.size(1);

    auto outHypotheses = torch::zeros({B, ransacHypotheses, 6}, torch::kFloat32);
    auto outScores = torch::zeros({B, ransacHypotheses, 1}, torch::kFloat32);
    auto outHypothesesSampledIndices = torch::zeros({B, ransacHypotheses, 4, 1}, torch::kFloat32);
    auto outInlierMaps = torch::ones({B, ransacHypotheses, MAX_PT_COUNT, 1}, torch::kInt32) * -1;
    auto outInitHypotheses = torch::zeros({B, ransacHypotheses, 6}, torch::kFloat32);

    // Get the valid 2D and 3D coordinates from the tensors
    dsacstar::vector3f coords2d_vector(B);
    dsacstar::vector3f coords3d_vector(B);
    std::vector<unsigned int> coords_size(B);
    #pragma omp parallel for
    for(int b_idx = 0; b_idx < B; b_idx++)
    {
        std::vector<std::vector<float>> b_pts_2d;
        std::vector<std::vector<float>> b_pts_3d;
        b_pts_2d.reserve(MAX_PT_COUNT);
        b_pts_3d.reserve(MAX_PT_COUNT);
        for(int pt_idx = 0; pt_idx < MAX_PT_COUNT; pt_idx++)
        {
            // Invalid 2D coordinates must be set to a value below -1000
            if(coords2d[b_idx][pt_idx][0] < -1000 && coords2d[b_idx][pt_idx][1] < -1000) break;

            b_pts_2d.push_back(std::vector<float>{coords2d[b_idx][pt_idx][0], coords2d[b_idx][pt_idx][1]});
            b_pts_3d.push_back(std::vector<float>{coords3d[b_idx][pt_idx][0], coords3d[b_idx][pt_idx][1], coords3d[b_idx][pt_idx][2]});
        }
        coords2d_vector[b_idx] = b_pts_2d;
        coords3d_vector[b_idx] = b_pts_3d;
        coords_size[b_idx] = b_pts_2d.size();
    }

    // Process each batch element separately
    for(int b_idx = 0; b_idx < B; b_idx++)
    {
        // Handle the case where the number of points are less than minPoints
        if (coords_size[b_idx] < minPoints) continue;

//        std::cout << "Batch: " <<  b_idx << std::endl;
        // Populate the camera matrix for the element in the batch
        cv::Mat_<float> camMat = cv::Mat_<float>::eye(3, 3);
        camMat(0, 0) = cameraMatrix[b_idx][0][0];
        camMat(1, 1) = cameraMatrix[b_idx][1][1];
        camMat(0, 2) = cameraMatrix[b_idx][0][2];
        camMat(1, 2) = cameraMatrix[b_idx][1][2];

//        std::cout << BLUETEXT("Sampling " << ransacHypotheses << " hypotheses.") << std::endl;
//        StopWatch stopW;

        // Sample RANSAC hypotheses
        std::vector<dsacstar::pose_t> hypotheses;
        std::vector<std::vector<int>> hypothesesSampledIndices;
        std::vector<std::vector<cv::Point2f>> pts2d;
        std::vector<std::vector<cv::Point3f>> pts3d;

//        std::cout << "Batch: " << b_idx << " Before sampling" << std::endl;
        dsacstar::sampleHypotheses(
                b_idx,
                coords2d_vector,
                coords3d_vector,
                coords_size,
                camMat,
                ransacHypotheses,
                MAX_HYPOTHESES_TRIES,
                inlierThreshold,
                hypotheses,
                hypothesesSampledIndices,
                pts2d,
                pts3d);

//          std::cout << "Batch: " << b_idx << " After sampling" << std::endl;

//        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

//        std::cout << BLUETEXT("Calculating scores.") << std::endl;

        // Compute reprojection error between the projected 2D points and the given 2D points
        std::vector<cv::Mat_<float>> reprojErrors(ransacHypotheses);
        cv::Mat_<double> jacobian_dummy;

        #pragma omp parallel for
        for(unsigned h = 0; h < hypotheses.size(); h++)
            reprojErrors[h] = dsacstar::getReproErrs(
                    b_idx,
                    coords2d_vector,
                    coords3d_vector,
                    coords_size,
                    hypotheses[h],
                    camMat,
                    maxReprojError,
                    jacobian_dummy);

//        std::cout << "Batch: " << b_idx << " After Reproj1" << std::endl;

//        std::cout << "After Repro Errors" << std::endl;

        // S oft inlier counting
        std::vector<double> scores = dsacstar::getHypScores(reprojErrors, inlierThreshold, inlierAlpha);

//        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//        std::cout << BLUETEXT("Drawing final hypothesis.") << std::endl;

        // Apply soft max to scores to get a distribution
        std::vector<double> hypProbs = dsacstar::softMax(scores);
//        double hypEntropy = dsacstar::entropy(hypProbs); // measure distribution entropy
//        int hypIdx = dsacstar::draw(hypProbs, true); // select winning hypothesis

//        std::cout << "Soft inlier count: " << scores[hypIdx] << " (Selection Probability: " << (int) (hypProbs[hypIdx]*100) << "%)" << std::endl;
//        std::cout << "Entropy of hypothesis distribution: " << hypEntropy << std::endl;
//
//        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
//        std::cout << BLUETEXT("Refining poses:") << std::endl;

        // Collect inliers and refine poses
        std::vector<dsacstar::pose_t> refHyps(ransacHypotheses);
        std::vector<cv::Mat_<int>> inlierMaps(ransacHypotheses);

        #pragma omp parallel for
        for(unsigned int h = 0; h < refHyps.size(); h++)
        {
            refHyps[h].first = hypotheses[h].first.clone();
            refHyps[h].second = hypotheses[h].second.clone();

            // Ignore probabilities that are hardly going to influence the training procedure
            if(hypProbs[h] < PROB_THRESH) continue;

            dsacstar::refineHyp(
                    b_idx,
                    coords2d_vector,
                    coords3d_vector,
                    coords_size,
                    reprojErrors[h],
                    camMat,
                    inlierThreshold,
                    MAX_REF_STEPS,
                    maxReprojError,
                    refHyps[h],
                    inlierMaps[h]);
        }
//        std::cout << "Batch: " << b_idx << " After refinement" << std::endl;
//        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

//std::cout << "End" << std::endl;

        // Write results to PyTorch accessor tensors
        #pragma omp parallel for
        for(int h = 0; h < refHyps.size(); h++)
        {
            // Hypotheses poses
            outHypotheses[b_idx][h][0] = (float)refHyps[h].first.at<double>(0, 0);
            outHypotheses[b_idx][h][1] = (float)refHyps[h].first.at<double>(1, 0);
            outHypotheses[b_idx][h][2] = (float)refHyps[h].first.at<double>(2, 0);
            outHypotheses[b_idx][h][3] = (float)refHyps[h].second.at<double>(0, 0);
            outHypotheses[b_idx][h][4] = (float)refHyps[h].second.at<double>(1, 0);
            outHypotheses[b_idx][h][5] = (float)refHyps[h].second.at<double>(2, 0);

            // Hypotheses scores
            outScores[b_idx][h][0] = scores[h];

            // Sampled indices of the hypotheses
            for(int pt_idx = 0; pt_idx < hypothesesSampledIndices[h].size(); pt_idx++)
            {
                outHypothesesSampledIndices[b_idx][h][pt_idx][0] = hypothesesSampledIndices[h][pt_idx];
            }

            // Inlier maps --> Tells which elements have been chosen using a boolean mask
            for(int pt_idx = 0; pt_idx < inlierMaps[h].rows; pt_idx++)
            {
                outInlierMaps[b_idx][h][pt_idx][0] = inlierMaps[h](pt_idx, 0);
            }

            // Init hypotheses
            outInitHypotheses[b_idx][h][0] = (float)hypotheses[h].first.at<double>(0, 0);
            outInitHypotheses[b_idx][h][1] = (float)hypotheses[h].first.at<double>(1, 0);
            outInitHypotheses[b_idx][h][2] = (float)hypotheses[h].first.at<double>(2, 0);
            outInitHypotheses[b_idx][h][3] = (float)hypotheses[h].second.at<double>(0, 0);
            outInitHypotheses[b_idx][h][4] = (float)hypotheses[h].second.at<double>(1, 0);
            outInitHypotheses[b_idx][h][5] = (float)hypotheses[h].second.at<double>(2, 0);

        }
//        std::cout << "Batch: " << b_idx <<  " After writing all the elements to PyTorch" << std::endl;
    }

    return {outHypotheses, outScores, outHypothesesSampledIndices, outInlierMaps, outInitHypotheses};
}

/**
 * @brief Performs pose estimation, and calculates the gradients of the pose loss wrt to scene coordinates.
 * @param sceneCoordinatesSrc Scene coordinate prediction, (1x3xHxW) with 1=batch dimension (only batch_size=1 supported atm), 3=scene coordainte dimensions, H=height and W=width.
 * @param outSceneCoordinatesGradSrc Scene coordinate gradients (output parameter). (1x3xHxW) same as scene coordinate input.
 * @param gtPoseSrc Ground truth camera pose, (4x4) tensor.
 * @param ransacHypotheses Number of RANSAC iterations.
 * @param inlierThreshold Inlier threshold for RANSAC in px.
 * @param focalLength Focal length of the camera in px.
 * @param ppointX Coordinate (X) of the prinicpal points.
 * @param ppointY Coordinate (Y) of the prinicpal points.
 * @param wLossRot Weight of the rotation loss term.
 * @param wLossTrans Weight of the translation loss term.
 * @param softClamp Use sqrt of pose loss after this threshold.
 * @param inlierAlpha Alpha parameter for soft inlier counting.
 * @param maxReproj Reprojection errors are clamped above this value (px).
 * @param subSampling Sub-sampling  of the scene coordinate prediction wrt the input image.
 * @param randomSeed External random seed to make sure we draw different samples across calls of this function.
 * @return DSAC expectation of the pose loss.
 */

torch::Tensor dsacstar_backward(
        const torch::Tensor &coords2dSrc,
        const torch::Tensor &coords3dSrc,
        const torch::Tensor &refinedHypothesesSrc,
        const torch::Tensor &scoresSrc,
        const torch::Tensor &initHypothesesSrc,
        const torch::Tensor &inlierMapsSrc,
        const torch::Tensor &hypothesesSampledIndicesSrc,
        const torch::Tensor &cameraMatrixSrc,
        const torch::Tensor &dLoss_dHypSrc,
        const torch::Tensor &dLoss_dScoresSrc,
        float inlierThreshold,
        float inlierAlpha,
        float maxReprojError,
        int minPoints)
{
    // Access to tensor objects
    dsacstar::coord_t coords2d = coords2dSrc.accessor<float, 3>();
    dsacstar::coord_t coords3d = coords3dSrc.accessor<float, 3>();
    dsacstar::coord_t refHypsTensor = refinedHypothesesSrc.accessor<float, 3>();
    dsacstar::coord_t scores = scoresSrc.accessor<float, 3>();
    dsacstar::coord_t initHypsTensor = initHypothesesSrc.accessor<float, 3>();
    auto inlierMapsTensor = inlierMapsSrc.accessor<int, 4>();
    auto hypothesesSampledIndicesTensor = hypothesesSampledIndicesSrc.accessor<float, 4>();
    dsacstar::coord_t cameraMatrix = cameraMatrixSrc.accessor<float, 3>();
    dsacstar::coord_t dLoss_dHypTensor = dLoss_dHypSrc.accessor<float, 3>();
    dsacstar::coord_t dLoss_dScoresTensor = dLoss_dScoresSrc.accessor<float, 3>();

    int B = refHypsTensor.size(0);
    int HYP_COUNT = refHypsTensor.size(1);
    int MAX_PT_COUNT = coords2d.size(1);

    // Tensors to send data back to PyTorch
    auto out_dLoss_dCoords2d = torch::zeros({B, MAX_PT_COUNT, 2}, torch::kFloat32);

    // Get the valid 2D and 3D coordinates from the tensors
    dsacstar::vector3f coords2d_vector(B);
    dsacstar::vector3f coords3d_vector(B);
    std::vector<unsigned int> coords_size(B);
    cv::Mat_<int> grad_mask = cv::Mat_<int>::zeros(B, MAX_PT_COUNT);
    #pragma omp parallel for
    for(int b_idx = 0; b_idx < B; b_idx++)
    {
        std::vector<std::vector<float>> b_pts_2d;
        std::vector<std::vector<float>> b_pts_3d;
        b_pts_2d.reserve(MAX_PT_COUNT);
        b_pts_3d.reserve(MAX_PT_COUNT);
        for(int pt_idx = 0; pt_idx < MAX_PT_COUNT; pt_idx++)
        {
            if(coords2d[b_idx][pt_idx][0] < -1000 && coords2d[b_idx][pt_idx][1] < -1000) break;

            b_pts_2d.push_back(std::vector<float>{coords2d[b_idx][pt_idx][0], coords2d[b_idx][pt_idx][1]});
            b_pts_3d.push_back(std::vector<float>{coords3d[b_idx][pt_idx][0], coords3d[b_idx][pt_idx][1], coords3d[b_idx][pt_idx][2]});
            grad_mask.at<int>(b_idx, pt_idx) = 1;
        }
        coords2d_vector[b_idx] = b_pts_2d;
        coords3d_vector[b_idx] = b_pts_3d;
        coords_size[b_idx] = b_pts_2d.size();
    }

    for(int b_idx = 0; b_idx < B; b_idx++)
    {
        // Handle the case where the number of points are less than minPoints
        if (coords_size[b_idx] < minPoints) continue;

        // Populate the camera matrix for the element in the batch
        cv::Mat_<float> camMat = cv::Mat_<float>::eye(3, 3);
        camMat(0, 0) = cameraMatrix[b_idx][0][0];
        camMat(1, 1) = cameraMatrix[b_idx][1][1];
        camMat(0, 2) = cameraMatrix[b_idx][0][2];
        camMat(1, 2) = cameraMatrix[b_idx][1][2];

        // Convert refHypsTensor, probs and inliers into cv::Mat_ objects
        std::vector<dsacstar::pose_t> refHyps(HYP_COUNT);
        std::vector<double> scores_vector(HYP_COUNT);
        std::vector<dsacstar::pose_t> initHyps(HYP_COUNT);
        std::vector<std::vector<int>> inlierMaps(HYP_COUNT);
        std::vector<std::vector<int>> hypSampledIndices(HYP_COUNT);
        std::vector<cv::Mat_<double>> dLoss_dHyp(HYP_COUNT);
        std::vector<double> dLoss_dScore(HYP_COUNT);

        #pragma omp parallel for
        for(unsigned int h = 0; h < HYP_COUNT; h++)
        {
            // Final refined hypotheses
            dsacstar::pose_t pose;
            pose.first = cv::Mat_<double>::zeros(3, 1);
            pose.second = cv::Mat_<double>::zeros(3, 1);
            pose.first.at<double>(0, 0) = refHypsTensor[b_idx][h][0];
            pose.first.at<double>(1, 0) = refHypsTensor[b_idx][h][1];
            pose.first.at<double>(2, 0) = refHypsTensor[b_idx][h][2];
            pose.second.at<double>(0, 0) = refHypsTensor[b_idx][h][3];
            pose.second.at<double>(1, 0) = refHypsTensor[b_idx][h][4];
            pose.second.at<double>(2, 0) = refHypsTensor[b_idx][h][5];
            refHyps[h] = pose;

            // Scores of the refined hypotheses
            scores_vector[h] = scores[b_idx][h][0];

            // Unrefined hypotheses
            dsacstar::pose_t init_pose;
            init_pose.first = cv::Mat_<double>::zeros(3, 1);
            init_pose.second = cv::Mat_<double>::zeros(3, 1);
            init_pose.first.at<double>(0, 0) = initHypsTensor[b_idx][h][0];
            init_pose.first.at<double>(1, 0) = initHypsTensor[b_idx][h][1];
            init_pose.first.at<double>(2, 0) = initHypsTensor[b_idx][h][2];
            init_pose.second.at<double>(0, 0) = initHypsTensor[b_idx][h][3];
            init_pose.second.at<double>(1, 0) = initHypsTensor[b_idx][h][4];
            init_pose.second.at<double>(2, 0) = initHypsTensor[b_idx][h][5];
            initHyps[h] = pose;

            // Inlier maps
            for(int pt_idx = 0; pt_idx < inlierMapsTensor.size(2); pt_idx++)
            {
                if(inlierMapsTensor[b_idx][h][pt_idx][0] > 0)
                {
                    inlierMaps[h].push_back(pt_idx);
                }
            }

            // Sampled indices of hypotheses
            for(int pt_idx = 0; pt_idx < hypothesesSampledIndicesTensor.size(2); pt_idx++)
            {
                hypSampledIndices[h].push_back(hypothesesSampledIndicesTensor[b_idx][h][pt_idx][0]);
            }

            // dLoss_dHyp
            cv::Mat_<double> grad_loss_hyp = cv::Mat_<double>::zeros(1, 6);
            grad_loss_hyp.at<double>(0, 0) = dLoss_dHypTensor[b_idx][h][0];
            grad_loss_hyp.at<double>(0, 1) = dLoss_dHypTensor[b_idx][h][1];
            grad_loss_hyp.at<double>(0, 2) = dLoss_dHypTensor[b_idx][h][2];
            grad_loss_hyp.at<double>(0, 3) = dLoss_dHypTensor[b_idx][h][3];
            grad_loss_hyp.at<double>(0, 4) = dLoss_dHypTensor[b_idx][h][4];
            grad_loss_hyp.at<double>(0, 5) = dLoss_dHypTensor[b_idx][h][5];
            dLoss_dHyp[h] = grad_loss_hyp;

            // dLoss_dScore
            dLoss_dScore[h] = dLoss_dScoresTensor[b_idx][h][0];
        }

        std::vector<double> hypProbs = dsacstar::softMax(scores_vector);

        // Path 1: Hypothesis Path --> Only on the inliers in the last iteration!
//        std::cout << BLUETEXT("Calculating j_dLoss_dCoords2d_wrtHyp_list wrt hypotheses.") << std::endl;
//        StopWatch stopW;

        // We want to compute the j_dLoss_dCoords2d_wrtHyp_list of the hypothesis wrt the image coordinates
        std::vector<cv::Mat_<double>> dHyp_dCoords2d(HYP_COUNT);
        std::vector<std::vector<cv::Mat_<double>>> dResidualNorm_dCoords2d_list(HYP_COUNT);
        std::vector<std::vector<cv::Mat_<double>>> dResidualNorm_dHyp_list(HYP_COUNT);

        #pragma omp parallel for
        for(unsigned int h = 0; h < HYP_COUNT; h++)
        {
            // Differentiate refinement around optimum found in last optimization iteration
            // Differential of the hypotheses wrt the 2D points
            dHyp_dCoords2d[h] = cv::Mat_<double>::zeros(6, MAX_PT_COUNT * 2);

            if(hypProbs[h] < PROB_THRESH) continue; // skip hypothesis with no impact on expectation

            // Collect inlier correspondences of last refinement iteration
            std::vector<cv::Point2f> imgPts_inliers;
            std::vector<cv::Point3f> objPts_inliers;
            std::vector<unsigned int> inlier_coords2d_index_map;

            for(unsigned int idx = 0; idx < inlierMaps[h].size(); idx++)
            {
                // The inlier map has already been filtered in when loading the data
                int pt_idx = (int) inlierMaps[h][idx];
                if(pt_idx < 0) continue;

                imgPts_inliers.emplace_back(coords2d_vector[b_idx][pt_idx][0], coords2d_vector[b_idx][pt_idx][1]);
                objPts_inliers.emplace_back(coords3d_vector[b_idx][pt_idx][0], coords3d_vector[b_idx][pt_idx][1], coords3d_vector[b_idx][pt_idx][2]);
                inlier_coords2d_index_map.push_back(pt_idx);
            }

            // Not enough inliers found in the inlier set. The gradients should be 0
            if(imgPts_inliers.size() < 4)
                continue;

            // We want to compute J_r = dResiduals_dHyp. The residual in this case is the norm of the error
            std::vector<cv::Point2f> projections;
            cv::Mat_<double> j_dProj_dHyp;
            cv::projectPoints(objPts_inliers, refHyps[h].first, refHyps[h].second, camMat, cv::Mat(), projections, j_dProj_dHyp);

//            std::cout << "Jacobian: " << j_dProj_dHyp.rows << " " << j_dProj_dHyp.cols << std::endl;

            // Jacobian matrix of derivatives of image points (projections) wrt the rotation and translation (Shape = 2N x 6) --> Collection of dProj_dHyp
            j_dProj_dHyp = j_dProj_dHyp.colRange(0, 6);

            // Assemble the jacobean of the refinement residuals (Collection of dResidualNorm_dHyp)
            cv::Mat_<double> j_dResidualNorm_dHyp = cv::Mat_<double>::zeros(objPts_inliers.size(), 6);  // J_r
            cv::Mat_<double> dResidualNorm_dProj(1, 2);  // Derivative of norm wrt the projection
            cv::Mat_<double> dResidualNorm_dHyp(1, 6);  // Derivative of norm wrt the hypothesis
            dResidualNorm_dCoords2d_list[h].resize(objPts_inliers.size());

            for(unsigned int inlier_idx = 0; inlier_idx < imgPts_inliers.size(); inlier_idx++)
            {
                double err = std::max(cv::norm(projections[inlier_idx] - imgPts_inliers[inlier_idx]), EPS);
                if(err > maxReprojError || std::isnan(err)) continue;

                // Derivative of residual norm wrt the projection (dResidualNorm_dProj)
                dResidualNorm_dProj(0, 0) = 1 / err * (projections[inlier_idx].x - imgPts_inliers[inlier_idx].x);
                dResidualNorm_dProj(0, 1) = 1 / err * (projections[inlier_idx].y - imgPts_inliers[inlier_idx].y);

                // Derivative of the norm wrt the hypothesis (dResidualNorm_dProj * dProj_dHyp)
                dResidualNorm_dHyp = dResidualNorm_dProj * j_dProj_dHyp.rowRange(2 * inlier_idx, 2 * inlier_idx + 2);
                dResidualNorm_dHyp.copyTo(j_dResidualNorm_dHyp.row(inlier_idx));
            }

            // Calculate the pseudo inverse multiplied with -1 --> -J_r^+
            j_dResidualNorm_dHyp = -(j_dResidualNorm_dHyp.t() * j_dResidualNorm_dHyp).inv(cv::DECOMP_SVD) * j_dResidualNorm_dHyp.t();

            double max_JacobianResiduals = dsacstar::getMax(j_dResidualNorm_dHyp);
            if(max_JacobianResiduals > 10) j_dResidualNorm_dHyp = 0; // clamping for stability

            // Compute the gradient of the residual norm wrt the Coords2d --> dResidualNorm_dCoords2d
            cv::Mat rot;
            cv::Rodrigues(refHyps[h].first, rot);
            dResidualNorm_dCoords2d_list[h].resize(objPts_inliers.size());
            for(unsigned int ptIdx = 0; ptIdx < objPts_inliers.size(); ptIdx++)
            {
                // Derivative of the projection norm wrt the object points
                cv::Mat_<double> dResidualNorm_dCoords2d = dsacstar::computeDResidualNormDCoords2d(imgPts_inliers[ptIdx], objPts_inliers[ptIdx], rot, refHyps[h].second, camMat, maxReprojError);
                cv::Mat_<double> dHyp_dCoords2d_OneEntry = j_dResidualNorm_dHyp.col(ptIdx) * dResidualNorm_dCoords2d;  // Eqn 31 in the paper  // 6x2

                // Get the index of the inliers in the original coords2d (i.e. index before inlier filtering)
                unsigned int coords2d_idx = inlier_coords2d_index_map[ptIdx];
                int dIdx = coords2d_idx * 2;
                dHyp_dCoords2d_OneEntry.copyTo(dHyp_dCoords2d[h].colRange(dIdx, dIdx + 2));
            }
        }

        // Combine j_dLoss_dCoords2d_wrtHyp_list per hypothesis. We get dLoss_dHyp from PyTorch and we compute dHyp_dCoords2d
        std::vector<cv::Mat_<double>> j_dLoss_dCoords2d_wrtHyp_list(refHyps.size());

        #pragma omp parallel for
        for(unsigned h = 0; h < refHyps.size(); h++)
        {
            if(hypProbs[h] < PROB_THRESH) continue;
            j_dLoss_dCoords2d_wrtHyp_list[h] = dLoss_dHyp[h] * dHyp_dCoords2d[h];
        }
//        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        // Path 2: Score Path --> On all the points (Both inliers and outliers in the last iteration)
//        std::cout << BLUETEXT("Calculating j_dLoss_dCoords2d_wrtHyp_list wrt scores.") << std::endl;

        // Compute reprojection error between the projected 2D points and the given 2D points
        std::vector<cv::Mat_<float>> reprojErrors(HYP_COUNT);
        std::vector<cv::Mat_<double>> j_dResidualNorm_dInitHyp(HYP_COUNT);

        #pragma omp parallel for
        for(unsigned int h = 0; h < refHyps.size(); h++)
            reprojErrors[h] = dsacstar::getReproErrs(
                    b_idx,
                    coords2d_vector,
                    coords3d_vector,
                    coords_size,
                    initHyps[h],
                    camMat,
                    maxReprojError,
                    j_dResidualNorm_dInitHyp[h],
                    true);


        // The jacobian matrix of dLoss_dCoords2d computed wrt score.
        std::vector<cv::Mat_<double>> j_dLoss_dCoords2d_wrtScore_list;
        dsacstar::dScore(
                b_idx,
                coords2d_vector,
                coords3d_vector,
                coords_size,
                hypSampledIndices,
                j_dLoss_dCoords2d_wrtScore_list,
                dLoss_dScore,
                refHyps,
                reprojErrors,
                j_dResidualNorm_dInitHyp,
                hypProbs,
                camMat,
                inlierAlpha,
                inlierThreshold,
                maxReprojError);

//        std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

        // Assemble full gradient tensor
        for(unsigned h = 0; h < refHyps.size(); h++)
        {
            if(hypProbs[h] < PROB_THRESH) continue;

            for(int pt_idx = 0; pt_idx < MAX_PT_COUNT; pt_idx++)
            {
                // Compute the gradients of only the non-buffer coordinates. The gradients of the buffer elements remain 0.
                if(grad_mask.at<int>(b_idx, pt_idx) > 0)
                {
                    out_dLoss_dCoords2d[b_idx][pt_idx][0] += hypProbs[h] * j_dLoss_dCoords2d_wrtHyp_list[h](0,2 * pt_idx) + j_dLoss_dCoords2d_wrtScore_list[h](0, 2 * pt_idx);
                    out_dLoss_dCoords2d[b_idx][pt_idx][1] += hypProbs[h] * j_dLoss_dCoords2d_wrtHyp_list[h](0,2 * pt_idx + 1) + j_dLoss_dCoords2d_wrtScore_list[h](0, 2 * pt_idx + 1);
                }
            }
        }
    }
    return {out_dLoss_dCoords2d};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &dsacstar_forward, "DSAC* forward");
	m.def("backward", &dsacstar_backward, "DSAC* backward");
}

#pragma clang diagnostic pop