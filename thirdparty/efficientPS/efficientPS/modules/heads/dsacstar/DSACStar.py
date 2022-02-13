import torch
import dsacstar

class DSACStar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points_2d, points_3d, k, ransac_hyp_count=64, inlier_thresh=5, inlier_alpha=100, max_reproj_err=20, min_points=20, random_seed=1305):

        # Apply softmax to the scores tensor to get the prob distribution
        if ctx is not None:
            pass
            # print("Pred Flow Forward Start.")
            # DSACStar.savePointsToFile(points_2d, points_3d, False)
        else:
            pass
            # print("GT Flow Forward Start.")
            # DSACStar.savePointsToFile(points_2d, points_3d, True)

        hyp_tensor, scores_tensor, hyp_sampled_indices_tensor, inlier_maps_tensor, init_hyp_tensor = dsacstar.forward(points_2d, points_3d, k, ransac_hyp_count, inlier_thresh, inlier_alpha, max_reproj_err, min_points, random_seed)
        if ctx is not None:
            ctx.save_for_backward(points_2d, points_3d, hyp_tensor, scores_tensor, init_hyp_tensor, inlier_maps_tensor, hyp_sampled_indices_tensor, k)
            ctx.inlier_thresh = inlier_thresh
            ctx.inlier_alpha = inlier_alpha
            ctx.max_reproj_err = max_reproj_err
            ctx.min_points = min_points
            # print("Pred Flow Forward Done.")
        else:
            pass
            # print("GT Flow Forward Done.")


        return hyp_tensor, scores_tensor

    @staticmethod
    def backward(ctx, dLoss_dHyp, dLoss_dScores):

        points_2d, points_3d, hyp_tensor, scores_tensor, init_hyp_tensor, inlier_maps_tensor, hyp_sampled_indices_tensor, k = ctx.saved_tensors

        # DSACStar.saveTensorsToFile(points_2d, points_3d, hyp_tensor, scores_tensor, init_hyp_tensor, inlier_maps_tensor, hyp_sampled_indices_tensor, dLoss_dHyp, dLoss_dScores)

        inlier_thresh = ctx.inlier_thresh
        inlier_alpha = ctx.inlier_alpha
        max_reproj_err = ctx.max_reproj_err
        min_points = ctx.min_points
        # print("Backward Start.")
        dLoss_dPoints2d = dsacstar.backward(points_2d, points_3d, hyp_tensor, scores_tensor, init_hyp_tensor, inlier_maps_tensor, hyp_sampled_indices_tensor, k, dLoss_dHyp, dLoss_dScores, inlier_thresh, inlier_alpha, max_reproj_err, min_points)
        # print("Backward Done.")
        print("Gradient", torch.sum(dLoss_dPoints2d != dLoss_dPoints2d))

        return dLoss_dPoints2d, None, None

    @staticmethod
    def savePointsToFile(points_2d, points_3d, is_gt=False):
        if is_gt:
            save_path = "/home/gosalan/Documents/miscellaneous/points_2d_gt.csv"
        else:
            save_path = "/home/gosalan/Documents/miscellaneous/points_2d_pred.csv"
        with open(save_path, 'w') as fp:
            for b_idx in range(points_2d.shape[0]):
                for pt_idx in range(points_2d.shape[1]):
                    fp.write("{} {} ".format(points_2d[b_idx][pt_idx][0], points_2d[b_idx][pt_idx][1]))
                fp.write("\n")

        if is_gt:
            save_path = "/home/gosalan/Documents/miscellaneous/points_3d_gt.csv"
        else:
            save_path = "/home/gosalan/Documents/miscellaneous/points_3d_pred.csv"
        with open(save_path, 'w') as fp:
            for b_idx in range(points_3d.shape[0]):
                for pt_idx in range(points_3d.shape[1]):
                    fp.write("{} {} {} ".format(points_3d[b_idx][pt_idx][0], points_3d[b_idx][pt_idx][1],
                                                points_3d[b_idx][pt_idx][2]))
                fp.write("\n")

    @staticmethod
    def saveTensorsToFile(points_2d, points_3d, hyp_tensor, scores_tensor, init_hyp_tensor, inlier_maps_tensor, hyp_sampled_indices_tensor, dLoss_dHyp, dLoss_dScores):
        with open("/home/gosalan/Documents/miscellaneous/points_2d_backward.csv", 'w') as fp:
            for b_idx in range(points_2d.shape[0]):
                for pt_idx in range(points_2d.shape[1]):
                    fp.write("{} {} ".format(points_2d[b_idx][pt_idx][0], points_2d[b_idx][pt_idx][1]))
                fp.write("\n")

        with open("/home/gosalan/Documents/miscellaneous/points_3d_backward.csv", 'w') as fp:
            for b_idx in range(points_3d.shape[0]):
                for pt_idx in range(points_3d.shape[1]):
                    fp.write("{} {} {} ".format(points_3d[b_idx][pt_idx][0], points_3d[b_idx][pt_idx][1],
                                                points_3d[b_idx][pt_idx][2]))
                fp.write("\n")

        with open("/home/gosalan/Documents/miscellaneous/hypotheses_backward.csv", 'w') as fp:
            for b_idx in range(hyp_tensor.shape[0]):
                for h_idx in range(hyp_tensor.shape[1]):
                    hyp = hyp_tensor[b_idx][h_idx]
                    fp.write("{} {} {} {} {} {} ".format(hyp[0], hyp[1], hyp[2], hyp[3], hyp[4], hyp[5]))
                fp.write("\n")

        with open("/home/gosalan/Documents/miscellaneous/scores_backward.csv", 'w') as fp:
            for b_idx in range(scores_tensor.shape[0]):
                for s_idx in range(scores_tensor.shape[1]):
                    scores = scores_tensor[b_idx][h_idx]
                    fp.write("{} ".format(scores[0]))
                fp.write("\n")

        with open("/home/gosalan/Documents/miscellaneous/init_hypotheses_backward.csv", 'w') as fp:
            for b_idx in range(init_hyp_tensor.shape[0]):
                for h_idx in range(init_hyp_tensor.shape[1]):
                    init_hyp = init_hyp_tensor[b_idx][h_idx]
                    fp.write("{} {} {} {} {} {} ".format(init_hyp[0], init_hyp[1], init_hyp[2], init_hyp[3], init_hyp[4], init_hyp[5]))
                fp.write("\n")

        with open("/home/gosalan/Documents/miscellaneous/inlier_maps_backward.csv", 'w') as fp:
            for b_idx in range(inlier_maps_tensor.shape[0]):
                inlier_str = ""
                for h_idx in range(inlier_maps_tensor.shape[1]):
                    for pt_idx in range(inlier_maps_tensor.shape[2]):
                        inlier_str += "{} ".format(inlier_maps_tensor[b_idx][h_idx][pt_idx][0])
                    # inlier_str += "| "
                fp.write(inlier_str)
                fp.write("\n")
            fp.flush()

        with open("/home/gosalan/Documents/miscellaneous/sampled_indices_backward.csv", 'w') as fp:
            for b_idx in range(hyp_sampled_indices_tensor.shape[0]):
                for h_idx in range(hyp_sampled_indices_tensor.shape[1]):
                    si = hyp_sampled_indices_tensor[b_idx][h_idx]
                    fp.write("{} {} {} {} ".format(si[0][0], si[1][0], si[2][0], si[3][0]))
                fp.write("\n")
            fp.flush()

        with open("/home/gosalan/Documents/miscellaneous/dLoss_dHyp_backward.csv", 'w') as fp:
            for b_idx in range(dLoss_dHyp.shape[0]):
                for h_idx in range(dLoss_dHyp.shape[1]):
                    hyp = dLoss_dHyp[b_idx][h_idx]
                    fp.write("{} {} {} {} {} {} ".format(hyp[0], hyp[1], hyp[2], hyp[3], hyp[4], hyp[5]))
                fp.write("\n")

        with open("/home/gosalan/Documents/miscellaneous/dLoss_dScores_backward.csv", 'w') as fp:
            for b_idx in range(dLoss_dScores.shape[0]):
                for s_idx in range(dLoss_dScores.shape[1]):
                    scores = dLoss_dScores[b_idx][h_idx]
                    fp.write("{} ".format(scores[0]))
                fp.write("\n")
