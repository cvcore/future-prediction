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

#pragma once

#include <omp.h>
#include "thread_rand.h"
#include "dsacstar_util_rgbd.h"

// makros for coloring console output
#define GREENTEXT(output) "\x1b[32;1m" << output << "\x1b[0m"
#define REDTEXT(output) "\x1b[31;1m" << output << "\x1b[0m"
#define BLUETEXT(output) "\x1b[34;1m" << output << "\x1b[0m"
#define YELLOWTEXT(output) "\x1b[33;1m" << output << "\x1b[0m"

#define EPS 0.00000001
#define PI 3.1415926

namespace dsacstar
{
	/**
	* @brief Calculate original image positions of a scene coordinate prediction.
	* @param outW Width of the scene coordinate prediction.
	* @param outH Height of the scene coordinate prediction.
	* @param subSampling Sub-sampling of the scene coordinate prediction wrt. to the input image.
	* @param shiftX Horizontal offset in case the input image has been shifted before scene coordinare prediction.
	* @param shiftY Vertical offset in case the input image has been shifted before scene coordinare prediction.
	* @return Matrix where each entry contains the original 2D image position.
	*/
	cv::Mat_<cv::Point2i> createSampling(
		unsigned outW, unsigned outH, 
		int subSampling, 
		int shiftX, int shiftY)
	{
		cv::Mat_<cv::Point2i> sampling(outH, outW);

		#pragma omp parallel for
		for(unsigned x = 0; x < outW; x++)
		for(unsigned y = 0; y < outH; y++)
		{
			sampling(y, x) = cv::Point2i(
				x * subSampling + subSampling / 2 - shiftX,
				y * subSampling + subSampling / 2 - shiftY);
		}

		return sampling;
	}

	/**
	* @brief Wrapper for OpenCV solvePnP.
	* Properly handles empty pose inputs.
	* @param objPts List of 3D scene points.
	* @param imgPts List of corresponding 2D image points.
	* @param camMat Internal calibration matrix of the camera.
	* @param distCoeffs Distortion coefficients.
	* @param rot Camera rotation (input/output), axis-angle representation.
	* @param trans Camera translation.
	* @param extrinsicGuess Whether rot and trans already contain an pose estimate.
	* @param methodFlag OpenCV PnP method flag.
	* @return True if pose estimation succeeded.
	*/
	inline bool safeSolvePnP(
		const std::vector<cv::Point3f>& objPts,
		const std::vector<cv::Point2f>& imgPts,
		const cv::Mat& camMat,
		const cv::Mat& distCoeffs,
		cv::Mat& rot,
		cv::Mat& trans,
		bool extrinsicGuess,
		int methodFlag)
	{
		if(rot.type() == 0) rot = cv::Mat_<double>::zeros(3, 1);
		if(trans.type() == 0) trans= cv::Mat_<double>::zeros(3, 1);

		if(!cv::solvePnP(
			objPts,
			imgPts,
			camMat,
			distCoeffs,
			rot,
			trans,
			extrinsicGuess,
			methodFlag))
		{
			rot = cv::Mat_<double>::zeros(3, 1);
			trans = cv::Mat_<double>::zeros(3, 1);
			return false;
		}

		return true;
	}

	/**
	* @brief Samples a set of RANSAC camera pose hypotheses using PnP
	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	* @param sampling Contains original image coordinate for each scene coordinate predicted.
	* @param camMat Camera calibration matrix.
	* @param ransacHypotheses RANSAC iterations.
	* @param maxTries Repeat sampling an hypothesis if it is invalid
	* @param inlierThreshold RANSAC inlier threshold in px.
	* @param hypotheses (output parameter) List of sampled pose hypotheses.
	* @param sampledPoints (output parameter) Corresponding minimal set for each hypotheses, scene coordinate indices.
	* @param pts2d_minimal (output parameter) Corresponding minimal set for each hypotheses, 2D image coordinates.
	* @param pts3d_minimal (output parameter) Corresponding minimal set for each hypotheses, 3D scene coordinates.
	*/
	inline void sampleHypotheses(
	    const int b_idx,
		const dsacstar::vector3f &coords2d_vector,
		const dsacstar::vector3f &coords3d_vector,
		const std::vector<unsigned int> &coords_size,
		const cv::Mat_<float> &camMat,
		int ransacHypotheses,
		unsigned int maxTries,
		float inlierThreshold,
		std::vector<dsacstar::pose_t> &hypotheses,
		std::vector<std::vector<int>> &sampledPoints,
		std::vector<std::vector<cv::Point2f>> &pts2d_minimal,
		std::vector<std::vector<cv::Point3f>> &pts3d_minimal)
	{
		// Keep track of the points each hypothesis is sampled from
		sampledPoints.resize(ransacHypotheses);     
		pts2d_minimal.resize(ransacHypotheses);
		pts3d_minimal.resize(ransacHypotheses);
		hypotheses.resize(ransacHypotheses);

//		std::cout << "Batch: " << b_idx << " sampleHypotheses: After resize" << std::endl;

		// Sample ransacHypotheses number of hypotheses
		#pragma omp parallel for
		for(unsigned int h = 0; h < ransacHypotheses; h++)
		{
//		    int wasted_tries = 0;
            for (unsigned int t = 0; t < maxTries; t++)
            {
                std::vector<cv::Point2f> projections;
                pts2d_minimal[h].clear();
                pts3d_minimal[h].clear();
                sampledPoints[h].clear();

                // Get the number of points in the 2D and 3D arrays
                int PT_COUNT = coords_size[b_idx];
//                std::cout << "After clear: " << PT_COUNT << std::endl;

//                 std::cout << "Try start" << std::endl;
                // Sample 4 coordinates and make the 2D and 3D coordinate array. This is fed as input to the PnP algorithm
                for (unsigned int j = 0; j < 4; j++)
                {
                    // Sample an index from the point coordinates. Keep sampling indices till you get a unique index
                    int rand_idx = irand(0, PT_COUNT);
                    while ((std::find(sampledPoints[h].begin(), sampledPoints[h].end(), rand_idx) != sampledPoints[h].end()) || (rand_idx >= PT_COUNT) || (rand_idx < 0))
                        rand_idx = irand(0, PT_COUNT);

                    // Get the 2D and 3D coordinates corresponding to the rand_idx
                    if(rand_idx < 0 || rand_idx > coords2d_vector[b_idx].size() || rand_idx > coords3d_vector[b_idx].size())
                    {
                        std::cout << "Invalid index: " << rand_idx << std::endl;
                    }

                    assert(b_idx < coords2d_vector.size() && "Batch Idx more than coords2d_vector length");
                    assert(b_idx < coords3d_vector.size() && "Batch Idx more than coords3d_vector length");
                    assert(rand_idx < coords2d_vector[b_idx].size() &&  "rand_idx more than coords2d_vector[b_idx] length");
                    assert(rand_idx < coords3d_vector[b_idx].size() && "rand_idx more than coords3d_vector[b_idx] length");

                    pts2d_minimal[h].push_back(cv::Point2f(coords2d_vector[b_idx][rand_idx][0], coords2d_vector[b_idx][rand_idx][1]));
                    pts3d_minimal[h].push_back(cv::Point3f(coords3d_vector[b_idx][rand_idx][0], coords3d_vector[b_idx][rand_idx][1], coords3d_vector[b_idx][rand_idx][2]));

                    // Save the sampled index
                    sampledPoints[h].push_back(rand_idx);
                }

                assert((pts2d_minimal[h].size() == pts3d_minimal[h].size()) && "pts_2d_minimal and pts_2d_minimal should be of equal length!");
//                std::cout << "Try end" << std::endl;

                // Solve the PnP equation using the 4 sampled points
//                std::cout << "Before PnP" << std::endl;
                bool pnp_success = dsacstar::safeSolvePnP(pts3d_minimal[h], pts2d_minimal[h], camMat, cv::Mat(), hypotheses[h].first, hypotheses[h].second, false, cv::SOLVEPNP_P3P);
//                std::cout << "After PnP" << std::endl;
                if (!pnp_success)
                {
//                    wasted_tries++;
                    continue;
                }

                // Check reconstruction. The 4 sampled points should be reconstructed perfectly
                cv::projectPoints(
                        pts3d_minimal[h],
                        hypotheses[h].first,
                        hypotheses[h].second,
                        camMat,
                        cv::Mat(),
                        projections);

                assert((pts2d_minimal[h].size() == projections.size()) && "pts_2d_minimal and projections should be of equal length!");

                bool foundOutlier = false;
                for (unsigned j = 0; j < pts2d_minimal[h].size(); j++)
                {
                    float err = cv::norm(pts2d_minimal[h][j] - projections[j]);
                    if (err < inlierThreshold && !std::isnan(err))
                    {
                        // Continue only if the err is less than the inlierThreshold and err is not a NaN
                        continue;
                    }

                    foundOutlier = true;
                    break;
                }

                if (foundOutlier)
                {
                    continue; // Try sampling the points again!
                }
                else
                {
                    break; // We have good enough points for the hypothesis. End this try.
                }

            }

//            std::cout << "Waster Tries: " << wasted_tries << std::endl;
        }
	}

	/**
	* @brief Samples a set of RANSAC camera pose hypotheses using Kabsch
	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	* @param camera coordinates Camera coordinates calculated from measured depth, same format and size as scene coordinates.
	* @param validPts A list of valid 2D image positions where camera coordinates / measured depth exists.
	* @param ransacHypotheses RANSAC iterations.
	* @param maxTries Repeat sampling an hypothesis if it is invalid
	* @param inlierThreshold RANSAC inlier threshold in px.
	* @param hypotheses (output parameter) List of sampled pose hypotheses.
	* @param sampledPoints (output parameter) Corresponding minimal set for each hypotheses, scene coordinate indices.
	* @param eyePts (output parameter) Corresponding minimal set for each hypotheses, 3D camera coordinates.
	* @param objPts (output parameter) Corresponding minimal set for each hypotheses, 3D scene coordinates.
	*/
//	inline void sampleHypothesesRGBD(
//		dsacstar::coord_t& sceneCoordinates,
//		dsacstar::coord_t& cameraCoordinates,
//		const std::vector<cv::Point2i>& validPts,
//		int ransacHypotheses,
//		unsigned maxTries,
//		float inlierThreshold,
//		std::vector<dsacstar::pose_t>& hypotheses,
//		std::vector<std::vector<cv::Point2i>>& sampledPoints,
//		std::vector<std::vector<cv::Point3f>>& eyePts,
//		std::vector<std::vector<cv::Point3f>>& objPts)
//	{
//		// keep track of the points each hypothesis is sampled from
//		sampledPoints.resize(ransacHypotheses);
//		eyePts.resize(ransacHypotheses);
//		objPts.resize(ransacHypotheses);
//		hypotheses.resize(ransacHypotheses);
//
//		// sample hypotheses
//		#pragma omp parallel for
//		for(unsigned h = 0; h < hypotheses.size(); h++)
//		for(unsigned t = 0; t < maxTries; t++)
//		{
//			int batchIdx = 0; // only batch size=1 supported atm
//
//			std::vector<cv::Point3f> dists;
//			eyePts[h].clear();
//			objPts[h].clear();
//			sampledPoints[h].clear();
//
//			for(int j = 0; j < 3; j++)
//			{
//				// 2D location in the subsampled image
//				int ptIdx = irand(0, validPts.size());
//				int x = validPts[ptIdx].x;
//				int y = validPts[ptIdx].y;
//
//				// 3D camera coordinate
//				eyePts[h].push_back(cv::Point3f(
//					cameraCoordinates[batchIdx][0][y][x],
//					cameraCoordinates[batchIdx][1][y][x],
//					cameraCoordinates[batchIdx][2][y][x]));
//				// 3D object (=scene) coordinate
//				objPts[h].push_back(cv::Point3f(
//					sceneCoordinates[batchIdx][0][y][x],
//					sceneCoordinates[batchIdx][1][y][x],
//					sceneCoordinates[batchIdx][2][y][x]));
//				// 2D pixel location in the subsampled image
//				sampledPoints[h].push_back(cv::Point2i(x, y));
//			}
//
//
//			kabsch(eyePts[h], objPts[h], hypotheses[h]);
//			transform(objPts[h], hypotheses[h], dists);
//
//
//			// check reconstruction, 3 sampled points should be reconstructed perfectly
//			bool foundOutlier = false;
//			for(unsigned j = 0; j < eyePts[h].size(); j++)
//			{
//				if(cv::norm(eyePts[h][j] - dists[j])*100 < inlierThreshold) //measure distance in centimeters
//					continue;
//				foundOutlier = true;
//				break;
//			}
//
//			if(foundOutlier)
//				continue;
//			else
//				break;
//		}
//	}

	/**
	* @brief Calculate soft inlier counts.
	* @param reprojErrors Image of reprojection error for each pose hypothesis.
	* @param inlierThreshold RANSAC inlier threshold.
	* @param inlierAlpha Alpha parameter for soft inlier counting.
	* @return List of soft inlier counts for each hypothesis.
	*/
	inline std::vector<double> getHypScores(
		const std::vector<cv::Mat_<float>>& reprojErrors,
		float inlierThreshold,
		float inlierAlpha)
	{
		std::vector<double> scores(reprojErrors.size(), 0);

		// beta parameter for soft inlier counting
		float inlierBeta = 5.f / inlierThreshold;

		#pragma omp parallel for
		for(unsigned h = 0; h < reprojErrors.size(); h++)
        {
		    for(int pt_idx = 0; pt_idx < reprojErrors[h].rows; pt_idx++)
            {
                double softThreshold = inlierBeta * (reprojErrors[h](pt_idx) - inlierThreshold);
//                double denom = ;
//                std::cout << reprojErrors[h](pt_idx) << " ";
                softThreshold = 1. / (1 + std::exp(-softThreshold));
                scores[h] += 1 - softThreshold;

//                if (std::abs(scores[h]) < EPS) scores[h] = EPS;
            }
//            std::cout << std::endl;
        }

		#pragma omp parallel for
		for(unsigned h = 0; h < reprojErrors.size(); h++)
		{
		    if (reprojErrors[h].rows == 0 || reprojErrors[h].cols == 0) scores[h] *= 0.;
		    else scores[h] *= inlierAlpha / reprojErrors[h].rows / reprojErrors[h].cols;

//		    if (std::abs(scores[h]) < EPS) scores[h] = EPS;
		}

//		for(unsigned h = 0; h < scores.size(); h++)
//		{
//		    std::cout << scores[h] << " ";
//		}
//		std::cout << std::endl;


		return scores;
	}

	/**
	* @brief Calculate image of reprojection errors.
	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	* @param hyp Pose hypothesis to calculate the errors for.
	* @param sampling Contains original image coordinate for each scene coordinate predicted.
	* @param camMat Camera calibration matrix.
	* @param maxReproj Reprojection errors are clamped to this maximum value.
	* @param j_dResidualNorm_dHyp Jacobean matrix with derivatives of the 6D pose wrt. the reprojection error (num pts x 6).
	* @param calcJ Whether to calculate the jacobean matrix or not.
	* @return Image of reprojection errors.
	*/
	cv::Mat_<float> getReproErrs(
	    const int b_idx,
		const dsacstar::vector3f &coords2d_vector,
		const dsacstar::vector3f &coords3d_vector,
		const std::vector<unsigned int> &coords_size,
		const dsacstar::pose_t& hyp,
		const cv::Mat& camMat,
		float maxReproj,
	  	cv::Mat_<double>& j_dResidualNorm_dHyp,
  		bool calcJ = false)
	{
//	    std::cout << "Function enter" << std::endl;
//	    std::cout << coords2d_vector.sizes() << std::endl;

	    int PTS_COUNT = coords_size[b_idx];
//	    std::cout << "PTS_COUNT" << PTS_COUNT << std::endl;
	    cv::Mat_<float> reprojErrors = cv::Mat_<float>::zeros(PTS_COUNT, 1);

//        std::cout << "Before Point Array Population" << std::endl;
		std::vector<cv::Point3f> pts3d;
		std::vector<cv::Point2f> projections;	
		std::vector<cv::Point2f> pts2d;

		for(int pt_idx = 0; pt_idx < PTS_COUNT; pt_idx++)
        {
		    pts2d.push_back(cv::Point2f(coords2d_vector[b_idx][pt_idx][0], coords2d_vector[b_idx][pt_idx][1]));
		    pts3d.push_back(cv::Point3f(coords3d_vector[b_idx][pt_idx][0], coords3d_vector[b_idx][pt_idx][1], coords3d_vector[b_idx][pt_idx][2]));
        }

		if(pts2d.empty() || pts3d.empty()) return reprojErrors;

//		std::cout << "Before Project Function" << std::endl;
	    if(!calcJ)
	    {
//	        std::cout << "Inside no calcJ" << std::endl;
			// project object coordinate into the image using the given pose
			cv::projectPoints(
                    pts3d,
                    hyp.first,
                    hyp.second,
                    camMat,
                    cv::Mat(),
                    projections);

            // Replace NaNs
	        cv::patchNaNs(projections, maxReproj);
		}
	    else
	    {
	        cv::Mat_<double> j_dProj_dHyp;
	        cv::projectPoints(
                    pts3d,
                    hyp.first,
                    hyp.second,
                    camMat,
                    cv::Mat(),
                    projections,
                    j_dProj_dHyp);

             // Replace Nans
             cv::patchNaNs(projections, maxReproj);

            j_dProj_dHyp = j_dProj_dHyp.colRange(0, 6);

	        //assemble the jacobean of the refinement residuals
	        j_dResidualNorm_dHyp = cv::Mat_<double>::zeros(PTS_COUNT, 6);
	        cv::Mat_<double> dResidualNorm_dProj(1, 2);
	        cv::Mat_<double> dResidualNorm_dHyp(1, 6);

	        for(unsigned int ptIdx = 0; ptIdx < PTS_COUNT; ptIdx++)
	        {
	            double err = std::max(cv::norm(projections[ptIdx] - pts2d[ptIdx]), EPS);
	            if(std::isnan(err) || err > maxReproj)
	             {
	                err = maxReproj;
	                continue;
	             }
//	            if(err > maxReproj)
//	                continue;

	            // Derivative of norm
	            dResidualNorm_dProj(0, 0) = 1 / err * (projections[ptIdx].x - pts2d[ptIdx].x);
                dResidualNorm_dProj(0, 1) = 1 / err * (projections[ptIdx].y - pts2d[ptIdx].y);

                dResidualNorm_dHyp = dResidualNorm_dProj * j_dProj_dHyp.rowRange(2 * ptIdx, 2 * ptIdx + 2);
	            dResidualNorm_dHyp.copyTo(j_dResidualNorm_dHyp.row(ptIdx));
	        }
	    }

//	    std::cout << "After Project Function" << std::endl;

		// Compute the reprojection error for each point in the input 2D array
//		std::cout << "Projection Size:" << projections.size() << " " << pts2d.size() << std::endl;
		for(int p = 0; p < projections.size(); p++)
		{
			cv::Point2f currPt = pts2d[p] - projections[p];
			float l = std::min((float) cv::norm(currPt), maxReproj);
			if(std::isnan(l)) l = maxReproj;
//			std::cout << l << " ";
			reprojErrors.at<float>(p, 0) = l;
		}
//		std::cout << std::endl;

//		std::cout << "Return ReproErrors" << std::endl;

		return reprojErrors;
	}

	/**
	 * @brief Calculate an image of 3D distance errors for between scene coordinates and camera coordinates, given a pose.
	 * @param hyp Pose estimate.
	 * @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	 * @param camera coordinates Camera coordinates calculated from measured depth, same format and size as scene coordinates.
	 * @param validPts A list of valid 2D image positions where camera coordinates / measured depth exists.
	 * @param maxDist Clamp distance error with this value.
	 * @return Image of reprojectiob errors.
	 */
//	cv::Mat_<float> get3DDistErrs(
//	  const dsacstar::pose_t& hyp,
//	  const dsacstar::coord_t& sceneCoordinates,
//	  const dsacstar::coord_t& cameraCoordinates,
//	  const std::vector<cv::Point2i>& validPts,
//	  float maxDist)
//	{
//		int imH = sceneCoordinates.size(2);
//		int imW = sceneCoordinates.size(3);
//		int batchIdx = 0;  // only batch size=1 supported atm
//
//	    cv::Mat_<float> distMap = cv::Mat_<float>::ones(imH, imW) * maxDist;
//
//	    std::vector<cv::Point3f> points3D;
//	    std::vector<cv::Point3f> transformed3D;
//	    std::vector<cv::Point3f> pointsCam3D;
//	    std::vector<cv::Point2f> sources2D;
//
//	    // collect 2D-3D correspondences
//	    for(unsigned i = 0; i < validPts.size(); i++)
//	    {
//	    	int x = validPts[i].x;
//	    	int y = validPts[i].y;
//
//			pointsCam3D.push_back(cv::Point3f(
//				cameraCoordinates[batchIdx][0][y][x],
//				cameraCoordinates[batchIdx][1][y][x],
//				cameraCoordinates[batchIdx][2][y][x]));
//
//	 		points3D.push_back(cv::Point3f(
//				sceneCoordinates[batchIdx][0][y][x],
//				sceneCoordinates[batchIdx][1][y][x],
//				sceneCoordinates[batchIdx][2][y][x]));
//	    }
//
//	    if(points3D.empty()) return distMap;
//
//	    // transform scene coordinates to camera coordinates
//	    transform(points3D, hyp, transformed3D);
//
//	    // measure 3D distance
//	    for(unsigned p = 0; p < transformed3D.size(); p++)
//	    {
//			cv::Point3f curPt = pointsCam3D[p] - transformed3D[p];
//			//measure distance in centimeters
//			float l = std::min((float) cv::norm(curPt)*100, maxDist);
//			distMap(validPts[p].y, validPts[p].x) = l;
//	    }
//
//	    return distMap;
//	}


	/**
	* @brief Refine a pose hypothesis by iteratively re-fitting it to all inliers.
	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	* @param reproErrs Original reprojection errors of the pose hypothesis, used to collect the first set of inliers.
	* @param sampling Contains original image coordinate for each scene coordinate predicted.
	* @param camMat Camera calibration matrix.
	* @param inlierThreshold RANSAC inlier threshold.
	* @param maxRefSteps Maximum refinement iterations (re-calculating inlier and refitting).
	* @param maxReproj Reprojection errors are clamped to this maximum value.
	* @param hypothesis (output parameter) Refined pose.
	* @param inlierMap (output parameter) 2D image indicating which scene coordinate are (final) inliers.
	*/
	inline void refineHyp(
        const int b_idx,
        const dsacstar::vector3f &coords2d_vector,
		const dsacstar::vector3f &coords3d_vector,
		const std::vector<unsigned int> &coords_size,
		const cv::Mat_<float> &reprojErrors,
		const cv::Mat_<float> &camMat,
		float inlierThreshold,
		unsigned int maxRefSteps,
		float maxReprojError,
		dsacstar::pose_t &hypothesis,
		cv::Mat_<int> &inlierMap)
	{
		cv::Mat_<float> localReprojErrors = reprojErrors.clone();

		// Refine as long as inlier count increases
		unsigned bestInliers = 4; 

		// Refine current hypothesis
		for(unsigned int rStep = 0; rStep < maxRefSteps; rStep++)
		{
			// Collect inliers
			std::vector<cv::Point2f> localPts2d;
			std::vector<cv::Point3f> localPts3d;
			cv::Mat_<int> localInlierMap = cv::Mat_<int>::zeros(localReprojErrors.rows, localReprojErrors.cols);

			int PTS_COUNT = coords_size[b_idx];
			for(int pt_idx = 0; pt_idx < PTS_COUNT; pt_idx++)
			{
				if(localReprojErrors(pt_idx, 0) < inlierThreshold)
				{
					localPts2d.push_back(cv::Point2f(coords2d_vector[b_idx][pt_idx][0], coords2d_vector[b_idx][pt_idx][1]));
					localPts3d.push_back(cv::Point3f(coords3d_vector[b_idx][pt_idx][0], coords3d_vector[b_idx][pt_idx][1], coords3d_vector[b_idx][pt_idx][2]));
					localInlierMap(pt_idx, 0) = 1;
				}
			}

			if(localPts2d.size() <= bestInliers)
            {
                // Converged or less than 4 elements have a reprojError less than inlierThreshold.
			    break;
            }
			// Else, continue with the refinement
			bestInliers = localPts2d.size();

			// Recalculate pose
			dsacstar::pose_t hypUpdate;
			hypUpdate.first = hypothesis.first.clone();
			hypUpdate.second = hypothesis.second.clone();

			bool pnp_success = dsacstar::safeSolvePnP(
			        localPts3d,
			        localPts2d,
			        camMat,
			        cv::Mat(),
			        hypUpdate.first,
			        hypUpdate.second,
			        true,
			        (localPts2d.size() > 4) ? cv::SOLVEPNP_ITERATIVE : cv::SOLVEPNP_P3P);

    		if(!pnp_success)
            {
                // Abort if PnP fails
    		    break;
            }

    		// Else, apply the refinement update
			hypothesis = hypUpdate;
			inlierMap = localInlierMap;

			// Recalculate pose errors with the refined hypothesis
			cv::Mat_<double> jacobeanDummy;
            localReprojErrors = dsacstar::getReproErrs(
				b_idx,
				coords2d_vector,
				coords3d_vector,
				coords_size,
				hypothesis,
				camMat,
				maxReprojError,
				jacobeanDummy);
		}			
	}

	/**
	* @brief Refine a pose hypothesis by iteratively re-fitting it to all inliers (RGB-D version).
	* @param sceneCoordinates Scene coordinate prediction (1x3xHxW).
	* @param camera coordinates Camera coordinates calculated from measured depth, same format and size as scene coordinates.
	* @param distErrs Original 3D distance errors of the pose hypothesis, used to collect the first set of inliers.
 	* @param validPts A list of valid 2D image positions where camera coordinates / measured depth exists.
	* @param inlierThreshold RANSAC inlier threshold in centimeters.
	* @param maxRefSteps Maximum refinement iterations (re-calculating inlier and refitting).
	* @param maxDist Clamp distance error with this value.
	* @param hypothesis (output parameter) Refined pose.
	* @param inlierMap (output parameter) 2D image indicating which scene coordinate are (final) inliers.
	*/
//	inline void refineHypRGBD(
//		dsacstar::coord_t& sceneCoordinates,
//		dsacstar::coord_t& cameraCoordinates,
//		const cv::Mat_<float>& distErrs,
//		const std::vector<cv::Point2i>& validPts,
//		float inlierThreshold,
//		unsigned maxRefSteps,
//		float maxDist,
//		dsacstar::pose_t& hypothesis,
//		cv::Mat_<int>& inlierMap)
//	{
//		cv::Mat_<float> localDistErrs = distErrs.clone();
//		int batchIdx = 0; // only batch size=1 supported atm
//
//		// refine as long as inlier count increases
//		unsigned bestInliers = 3;
//
//		// refine current hypothesis
//		for(unsigned rStep = 0; rStep < maxRefSteps; rStep++)
//		{
//			// collect inliers
//			std::vector<cv::Point3f> localEyePts;
//			std::vector<cv::Point3f> localObjPts;
//			cv::Mat_<int> localInlierMap = cv::Mat_<int>::zeros(localDistErrs.size());
//
//			for(unsigned ptIdx = 0; ptIdx < validPts.size(); ptIdx++)
//			{
//				int x = validPts[ptIdx].x;
//				int y = validPts[ptIdx].y;
//
//				if(localDistErrs(y, x) < inlierThreshold)
//				{
//					localObjPts.push_back(cv::Point3f(
//						sceneCoordinates[batchIdx][0][y][x],
//						sceneCoordinates[batchIdx][1][y][x],
//						sceneCoordinates[batchIdx][2][y][x]));
//					localEyePts.push_back(cv::Point3f(
//						cameraCoordinates[batchIdx][0][y][x],
//						cameraCoordinates[batchIdx][1][y][x],
//						cameraCoordinates[batchIdx][2][y][x]));
//					localInlierMap(y, x) = 1;
//				}
//			}
//
//			if(localEyePts.size() <= bestInliers)
//				break; // converged
//			bestInliers = localEyePts.size();
//
//			// recalculate pose
//			dsacstar::pose_t hypUpdate;
//			hypUpdate.first = hypothesis.first.clone();
//			hypUpdate.second = hypothesis.second.clone();
//
//			kabsch(localEyePts, localObjPts, hypUpdate);
//
//			hypothesis = hypUpdate;
//			inlierMap = localInlierMap;
//
//			// recalculate pose errors
//			localDistErrs = dsacstar::get3DDistErrs(
//				hypothesis,
//				sceneCoordinates,
//				cameraCoordinates,
//				validPts,
//				maxDist);
//		}
//	}

	/**
	* @brief Applies soft max to the given list of scores.
	* @param scores List of scores.
	* @return Soft max distribution (sums to 1)
	*/
	std::vector<double> softMax(const std::vector<double>& scores)
	{
		double maxScore = 0;
		for(unsigned i = 0; i < scores.size(); i++)
			if(i == 0 || scores[i] > maxScore) maxScore = scores[i];

		std::vector<double> sf(scores.size());
		double sum = 0.0;

		for(unsigned i = 0; i < scores.size(); i++)
		{
			sf[i] = std::exp(scores[i] - maxScore);
			sum += sf[i];
		}
		for(unsigned i = 0; i < scores.size(); i++)
		{
			sf[i] /= sum;
		}

		return sf;
	}

	/**
	* @brief Calculate the Shannon entropy of a discrete distribution.
	* @param dist Discrete distribution. Probability per entry, should sum to 1.
	* @return  Shannon entropy.
	*/
	double entropy(const std::vector<double>& dist)
	{
		double e = 0;
		for(unsigned i = 0; i < dist.size(); i++)
			if(dist[i] > 0)
				e -= dist[i] * std::log2(dist[i]);

		return e;
	}

	/**
	* @brief Sample a hypothesis index.
	* @param probs Selection probabilities.
	* @param training If false, do not sample, but take argmax.
	* @return Hypothesis index.
	*/
	int draw(const std::vector<double>& probs, bool training)
	{
		std::map<double, int> cumProb;
		double probSum = 0;
		double maxProb = -1;
		double maxIdx = 0; 

		for(unsigned idx = 0; idx < probs.size(); idx++)
		{
			if(probs[idx] < EPS) continue;

			probSum += probs[idx];
			cumProb[probSum] = idx;

			if(maxProb < 0 || probs[idx] > maxProb)
			{
				maxProb = probs[idx];
				maxIdx = idx;
			}
		}

		if(training)
			return cumProb.upper_bound(drand(0, probSum))->second;
		else
			return maxIdx;
	}

	/**
	* @brief Transform scene pose (OpenCV format) to camera transformation, related by inversion.
	* @param pose Scene pose in OpenCV format (i.e. axis-angle and translation).
	* @return Camera transformation matrix (4x4).
	*/
	dsacstar::trans_t pose2trans(const dsacstar::pose_t& pose)
	{
		dsacstar::trans_t rot, trans = dsacstar::trans_t::eye(4, 4);
		cv::Rodrigues(pose.first, rot);

		rot.copyTo(trans.rowRange(0,3).colRange(0,3));
		trans(0, 3) = pose.second.at<double>(0, 0);
		trans(1, 3) = pose.second.at<double>(1, 0);
		trans(2, 3) = pose.second.at<double>(2, 0);

		return trans.inv(); // camera transformation is inverted scene pose
	}

	/**
	* @brief Transform camera transformation to scene pose (OpenCV format), related by inversion.
	* @param trans Camera transformation matrix (4x4)
	* @return Scene pose in OpenCV format (i.e. axis-angle and translation).
	*/
	dsacstar::pose_t trans2pose(const dsacstar::trans_t& trans)
	{
		dsacstar::trans_t invTrans = trans.inv();

		dsacstar::pose_t pose;
		cv::Rodrigues(invTrans.colRange(0,3).rowRange(0,3), pose.first);

		pose.second = cv::Mat_<double>(3, 1);
		pose.second.at<double>(0, 0) = invTrans(0, 3);
		pose.second.at<double>(1, 0) = invTrans(1, 3);
		pose.second.at<double>(2, 0) = invTrans(2, 3);

		return pose; // camera transformation is inverted scene pose
	}

	/**
	 * @brief Calculate the average of all matrix entries.
	 * @param mat Input matrix.
	 * @return Average of entries.
	 */
	double getAvg(const cv::Mat_<double>& mat)
	{
	    double avg = 0;
	    int count = 0;
	    
	    for(int x = 0; x < mat.cols; x++)
	    for(int y = 0; y < mat.rows; y++)
	    {
	    	double entry = std::abs(mat(y, x));
			if(entry > EPS)
			{
				avg += entry;
				count++;
			}
	    }
	    
	    return avg / (EPS + count);
	}

	/**
	 * @brief Return the maximum entry of the given matrix.
	 * @param mat Input matrix.
	 * @return Maximum entry.
	 */
	double getMax(const cv::Mat_<double>& mat)
	{
	    double m = -1;
	    
	    for(int x = 0; x < mat.cols; x++)
	    for(int y = 0; y < mat.rows; y++)
	    {
			double val = std::abs(mat(y, x));
			if(m < 0 || val > m)
			  m = val;
	    }
	    
	    return m;
	}

	/**
	 * @brief Return the median of all entries of the given matrix.
	 * @param mat Input matrix.
	 * @return Median entry.
	 */
	double getMed(const cv::Mat_<double>& mat)
	{
	    std::vector<double> vals;
	    
	    for(int x = 0; x < mat.cols; x++)
	    for(int y = 0; y < mat.rows; y++)
	    {
	    	double entry = std::abs(mat(y, x));
	    	if(entry > EPS) vals.push_back(entry);
	    }

	    if(vals.empty()) 
	    	return 0;

	    std::sort(vals.begin(), vals.end());
	    
	    return vals[vals.size() / 2];
	}	
}
