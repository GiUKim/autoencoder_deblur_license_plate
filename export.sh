
ls -alrt ./checkpoint
echo "input your model(.pt) file"
read torchmodel
echo "input width"
read width
echo "input height"
read height
echo "is gray scale? (input "g"[ray] or "c"[olor])"
read color
echo "exporting torch -> onnx ... "
python3 export.py $torchmodel $width $height $color
echo "Done"
echo "exporting onnx -> optimized onnx ... "
python3 export_unoptimized_layer.py
echo "Done"

