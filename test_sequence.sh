
CUDA_VISIBLE_DEVICES=3 python test_sequence.py   --result_path  './results/refsrrnn_cuf_siren_adists_allstage_only_future_t2/REDS_val_X2'   --space_scale  "2,2"  --data_path '/data1/shangwei/dataset/video/REDS/val/val_sharp'
CUDA_VISIBLE_DEVICES=3 python test_sequence.py   --result_path  './results/refsrrnn_cuf_siren_adists_allstage_only_future_t2/REDS_val_X3'   --space_scale  "3,3"  --data_path '/data1/shangwei/dataset/video/REDS/val/val_sharp'
CUDA_VISIBLE_DEVICES=3 python test_sequence.py   --result_path  './results/refsrrnn_cuf_siren_adists_allstage_only_future_t2/REDS_val_X4'   --space_scale  "4,4"  --data_path '/data1/shangwei/dataset/video/REDS/val/val_sharp'
CUDA_VISIBLE_DEVICES=3 python test_sequence.py   --result_path  './results/refsrrnn_cuf_siren_adists_allstage_only_future_t2/REDS_val_X6'   --space_scale  "6,6"  --data_path '/data1/shangwei/dataset/video/REDS/val/val_sharp'
CUDA_VISIBLE_DEVICES=3 python test_sequence.py   --result_path  './results/refsrrnn_cuf_siren_adists_allstage_only_future_t2/REDS_val_X8'   --space_scale  "8,8"  --data_path '/data1/shangwei/dataset/video/REDS/val/val_sharp'
#

CUDA_VISIBLE_DEVICES=3 python test_sequence.py   --result_path  './results/refsrrnn_cuf_siren_adists_allstage_only_future_t2/Vid4_val_X4'   --space_scale  "4,4"  --data_path  "/data1/shangwei/dataset/video/Vid4_val/Vid4"
CUDA_VISIBLE_DEVICES=3 python test_sequence.py   --result_path  './results/refsrrnn_cuf_siren_adists_allstage_only_future_t2/Vid4_val_X2.5_3.5'   --space_scale  "2.5,3.5"  --data_path  "/data1/shangwei/dataset/video/Vid4_val/Vid4"
CUDA_VISIBLE_DEVICES=3 python test_sequence.py   --result_path  './results/refsrrnn_cuf_siren_adists_allstage_only_future_t2/Vid4_val_X7.2_6'   --space_scale  "7.2,6"  --data_path  "/data1/shangwei/dataset/video/Vid4_val/Vid4"
