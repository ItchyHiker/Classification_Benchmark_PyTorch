python -m onnxsim girl_hair_mobilenetv2.onnx girl_hair_mobilenetv2.onnx
echo "Finished simplified"
/home/idealabs/Libs/ncnn/build/tools/onnx/onnx2ncnn girl_hair_mobilenetv2.onnx girl_hair_mobilenetv2.param girl_hair_mobilenetv2.bin
