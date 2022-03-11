# # ---------------------------------------------------------------------------
#   Author: Rumana Aktar 
#   Date: 03/10/2022
#   
#   Problem: from the PREICTION, generates shot_boundary.txt from a new dataset 
# 
#   For more information, contact:
#       Rumana Aktar
#       226 Naka Hall (EBW)
#       University of Missouri-Columbia
#       Columbia, MO 65211
#       rayy7@mail.missouri.edu
# # ---------------------------------------------------------------------------

import numpy as np, copy
import matplotlib.pyplot as plt


def getShotFromClass(sb_index, no_frames, filename, outdir, smoothing_pct, min_shot_length):
    sb_index = np.array(sb_index, int)    
    ones = np.ones((1, len(sb_index)), dtype=int)
    sb = np.zeros((1, no_frames), dtype=int)
    sb[0, sb_index] = ones

    
    # # --------------------------------------------------------------------------------------------------------------------------
    # if i-1 and i+1 are shot frames, force i to be shot frame
    sb2 = copy.deepcopy(sb)
    for i in range(1, no_frames-1):
        if sb[0,i-1] == 1 and sb[0,i+1] == 1:
            sb2[0,i] = 1


    # # --------------------------------------------------------------------------------------------------------------------------
    # if (i-1 or i-2 or i-3) and (i+1 or i+2 or i+3) are shot frames, force i to be shot frame
    sb3 = copy.deepcopy(sb2)
    for i in range(3, no_frames-3):
        if (sb2[0,i-3] == 1  or sb2[0,i-2] == 1 or sb2[0,i-1] == 1 )  and (sb2[0,i+3] == 1  or sb2[0,i+2] == 1 or sb2[0,i+1] == 1 ):
            sb3[0,i] = 1

    # # --------------------------------------------------------------------------------------------------------------------------
    sb = np.where(sb3[0, :] == 1)[0]    
    
    # print(sb); ccontinous boundaries
    # 70   93   94   95  393 1298 1299 1300 1301 1302 1303 1304 1305 1306 1370 1371 1372 1373 1374 1375 1376 1377 1378 1379 1380 1381 1382 1383
    # 1384 1385 1386 1387 1388 1389 1390 1391 1392 1393 1394 1395 1396 1397 1398 1399 1400 1401 1402 1403 1404 1405 1406 1407 1408 1409 1410 1411
    # 1412 1413 1414 1415 1416 1417 1418 1419 1420 1421 1422 1423 1424 1425 1426 1427 1428 1429 1430 1431 1432 1433 1434 1435 1444 1445 1446 1447
    # 2080 2154 2909 2998 2999 4055 5090 5091 5092 6331 6332 6333 6334 6335 6336 6337 6338 6339 6340 6341 6342 6343 6344 6345 6346 6347 6348 6349
    # 6350 6351 6352 6353 6354 6355 6356 6357 6358 6359 6360 6361 6362 6363 6364 6365 6366 6367 6368 6369 6370 6371 6372 6373 6374 6375 6376 6377
    # 6378 6379 6380 6381 6382 6383 6384 6385 6386 6387 6388 6389 6390 6391 6392 6393 6394 6395 6396 6397 6398 6399 6400 6401 6402 6403 6404 6405
    # 6406 6407 6408 6409 6410 6411 6412 6413 6414 6415 6416 6417 6418 6419 6420 6421 6422 6423 6424 6425 6426 8700 8701 8702 8703 8704 8705 8706
    # 8707 8708 8709 8710 8711 8712 8713 8714 8715 8716 8717 8718 8719 8720 8721 8722 8723 8724 8725 8726 8727 8728 8729 8730 8731 8732 8733 8734
    # 8735 8736 8737 8738 8739 8740 8741 8742 8743 9090 9279 9280 9281 9282 9283 9284 9285 9286 9287 9288 9289 9290]
    
    # # -----------------------------------------------Get the borders without any modification----------------------------------------------------------------
    shots, bdrs = get_shots_borders(sb, 0, 0)   # smoothing_pct, min_shot_length = 0, 0


    # # -----------------------------------------------now remove false positive----------------------------------------------------------------
    histDiff = np.loadtxt(filename, dtype=float)
    mean_histDiff = np.mean(histDiff)

    sb_copy = set(sb)
    for bd_st, bd_nd, bd_len in bdrs:
        bd_histDiff_avg = sum( histDiff[bd_st: bd_nd+1]) / bd_len
        
        # # ------ if boundary_histDiff_avg < mean_histDiff * 2.5, that means it is NOT a boundary frame, remove it
        if bd_histDiff_avg < mean_histDiff * 2.5:
            for x in range(bd_st, bd_nd + 1):
                sb_copy.remove(x)


    # # --------- sort the new_bd and convert to np.array
    new_bd = list(sb_copy);     new_bd.sort();    new_bd = np.array(new_bd, int)
    
    
    # -----------------------------------------------Get the NEW borders with minimum shot_length and smoothing_pct----------------------------------------------------------------
    shots, bdrs = get_shots_borders(new_bd, smoothing_pct, min_shot_length)
    
    # # -----------------------------------------------Save the shots and border information----------------------------------------------------------------
    np.savetxt( outdir + 'shot_boundary.txt', np.array(shots), fmt='%i', delimiter=",")
    np.savetxt( outdir + 'border_boundary.txt', np.array(bdrs), fmt='%i', delimiter=",")



def get_shots_borders(sb, smoothing_pct, min_shot_length):

    shots, bdrs = [], []

    shot_starts, i = 0, 0
    while i < len(sb)-1:


        # get shot_ends here; also border_starts, border_ends
        shot_ends, bdrs_starts, bdrs_ends = sb[i]-1, sb[i], sb[i]        
        while i < len(sb)-1 and sb[i+1]-sb[i] == 1:
            i += 1
        bdrs_ends=sb[i]

        shot_length = shot_ends-shot_starts+1

        # keep all the shots which length is atleast 75 before compression around boundary
        if shot_length >= min_shot_length:
            shot_reduction = (shot_length * smoothing_pct ) // 100
            shot_reduction= min(shot_reduction, 20)
            if shot_starts != 0:
                shot_starts_new = shot_starts + shot_reduction
            else:       shot_starts_new = shot_starts

            shot_ends_new = shot_ends - shot_reduction
            shots.append([shot_starts_new, shot_ends_new, shot_ends_new-shot_starts_new+1])
        
        bdrs.append([bdrs_starts, bdrs_ends, bdrs_ends-bdrs_starts+1])
        shot_starts = sb[i]+1
        i = i+1

    return [shots, bdrs]


