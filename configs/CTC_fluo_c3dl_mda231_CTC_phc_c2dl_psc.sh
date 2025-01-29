cnt: '/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/Fluo-C3DL-MDA231_slice9/01/'
# cnt: '/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/BF-C2DL-HSC_backRemoved/01/'
source_type: 'CTC' #choices=['liveCell', 'CTC', 'cellIm']
cnt_zoom: 0.65 #zoom for the content images zoomout if <1
masks_path: '/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/Fluo-C3DL-MDA231_slice9/01_ST/SEG/' #masks for the content images
sty: '/netshares/BiomedicalImageAnalysis/Resources/CellTrackingChallenge_UlmanNMeth/2024/PhC-C2DL-PSC/01/'
# sty_zoom: 0.0 #zoom for the style images, no need for this
ddim_inv_steps: 50
save_feat_steps: 50
start_step: 49
ddim_eta: 0.0
H: 512
W: 512
C: 4
f: 8
T: 3.0
gamma: 0.75
attn_layer: '6,7,8,9,10,11'
model_config: 'models/ldm/stable-diffusion-v1/v1-inference.yaml'
precomputed: ''
ckpt: 'models/ldm/stable-diffusion-v1/sd-v1-4.ckpt'
precision: 'autocast'
output_path: '/work/scratch/yilmaz/transferred_styles/CTC_fluoC3dlMda231_zoom0.65_CTCPhcC2dlPsc/'
without_init_adain: false
without_attn_injection: false
fromMask: false