config.py -> hyperparams, path 수정 후 train.py 실행

[cavalry bin 변환]

./export.sh 
pt model 선택 후 outcf.onnx로 최종 변환

output onnx로 cavalry 변환 환경에서 
pip3 install -U pip && pip3 install onnx-simplifier  설치 후 (최초 1번)
onnxsim ./outcf.onnx ./outcf.onnx  (onnxsim [input] [output])
으로 layer simplification 

.config 설정 후 make run_mode=cavalry 


---

# visualize 안되면

pip uninstall opencv-python-headless
pip install opencv-python
pip install opencv-contrib-python

##

