import numpy as np
import cv2.ximgproc as xip
import cv2

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    Il_pad = cv2.copyMakeBorder(Il, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    Ir_pad = cv2.copyMakeBorder(Ir, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    Il_binary = np.empty((h, w, 8, ch), dtype=bool)
    Ir_binary = np.empty((h, w, 8, ch), dtype=bool)
    Il_cost = np.empty((h, w, max_disp+1), dtype=np.float32)
    Ir_cost = np.empty((h, w, max_disp+1), dtype=np.float32)
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    # mask = [0, 1, 2, 5, 8, 7, 6, 3]
    # for y in range(1, h+1):
    #     for x in range(1, w+1):
    #         Il_wdw = Il_pad[y-1:y+2, x-1:x+2] > Il_pad[y, x]
    #         Ir_wdw = Ir_pad[y-1:y+2, x-1:x+2] > Ir_pad[y, x]
    #         Il_binary[y-1, x-1] = np.reshape(Il_wdw, (9, ch))[mask]
    #         Ir_binary[y-1, x-1] = np.reshape(Ir_wdw, (9, ch))[mask]

    pos = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (y, x) in enumerate(pos):
        Il_binary[:,:,i] = (np.roll(Il_pad, (y, x), axis=(1, 0)) > Il_pad)[1:h+1, 1:w+1]
        Ir_binary[:,:,i] = (np.roll(Ir_pad, (y, x), axis=(1, 0)) > Ir_pad)[1:h+1, 1:w+1]

    for d in range(max_disp+1):
        Il_binary_shifted = Il_binary[:, d:]
        Ir_binary_shifted = Ir_binary[:, :w-d]
        hamming = np.sum(np.logical_xor(Il_binary_shifted, Ir_binary_shifted), axis=(2, 3))
        Il_cost[:,:,d] = np.concatenate((np.reshape(np.repeat(hamming[:,0], d), (h, d)), hamming), axis=1)
        Ir_cost[:,:,d] = np.concatenate((hamming, np.reshape(np.repeat(hamming[:,-1], d), (h, d))), axis=1)
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for d in range(max_disp+1):
        Il_cost[:,:,d] = xip.jointBilateralFilter(Il, Il_cost[:,:,d], -1, 4, 5)
        Ir_cost[:,:,d] = xip.jointBilateralFilter(Ir, Ir_cost[:,:,d], -1, 4, 5)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    Il_winner = np.argmin(Il_cost, axis=2)
    Ir_winner = np.argmin(Ir_cost, axis=2)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    lr_consistency = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            if(Il_winner[y, x] != Ir_winner[y, x - Il_winner[y, x]]):
                lr_consistency[y, x] = 1

    left_fill_hole = cv2.copyMakeBorder(Il_winner, 0, 0, 1, 1, cv2.BORDER_REPLICATE)
    right_fill_hole = cv2.copyMakeBorder(Il_winner, 0, 0, 1, 1, cv2.BORDER_REPLICATE)

    for y in range(h):
        for x in range(1, w+1):
            if(lr_consistency[y, x-1]):          
                left_fill_hole[y, x] = left_fill_hole[y, x-1]
        for x in range(w, 0, -1):
            if(lr_consistency[y, x-1]):          
                right_fill_hole[y, x] = right_fill_hole[y, x+1]
    final = np.minimum(left_fill_hole, right_fill_hole)

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), final.astype(np.uint8), r=17)
    return labels.astype(np.uint8)