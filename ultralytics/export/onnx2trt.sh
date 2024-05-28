
onnx_path="onnx/suspension_post_nms.onnx"
save_trt_path="trt_model/suspension.plan"

trtexec --onnx=$onnx_path --saveEngine=$save_trt_path --workspace=3000 --verbose
