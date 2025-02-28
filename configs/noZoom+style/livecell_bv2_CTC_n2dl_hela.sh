cnt: '/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/images/livecell_train_val_images/bv2_backRemoved/'
# cnt: '/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/BF-C2DL-HSC_backRemoved/01/'
source_type: 'liveCell' #choices=['liveCell', 'CTC', 'cellIm']
cnt_zoom: 0.0 #zoom for the content images
masks_path: '/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/livecell/GTmasks_bv2/instance/' #masks for the content images
sty: '/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/Fluo-N2DL-HeLa/01/'
# sty_zoom: 0 #zoom for the style images
ddim_inv_steps: 50
save_feat_steps: 50
start_step: 49
ddim_eta: 0.0
H: 512
W: 512
C: 4
f: 8
T: 1.2
gamma: 0.75
attn_layer: '6,7,8,9,10,11'
model_config: 'models/ldm/stable-diffusion-v1/v1-inference.yaml'
precomputed: ''
ckpt: 'models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
precision: 'autocast'
output_path: '/work/scratch/yilmaz/transferred_styles/noZoom+style/livecell_bv2_CTC_n2dl_hela/'
without_init_adain: false
without_attn_injection: false
fromMask: false
rgb: false