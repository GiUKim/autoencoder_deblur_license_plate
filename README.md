## origin Keras source from [ https://github.com/inzapp/sigmoid-classifier ]

*** 
Modify .config file

TRAIN : python train.py

FORWARDING TEST : python predict.py

EXPORT ONNX : ./export.sh 

***
## .config file controller

1. TRAIN_PATH : Train dataset absolute path
2. VAL_PATH : Validation dataset absolute path
3. CHECKPOINT_PATH : Path for save trained model weights
4. PRETRAINED_MODEL_PATH : Path of pre-trained model for continue training. if empty, None
5. PREDICT_SOURCE_PATH : Path of images(.jpg) for predict
6. PREDICT_DESTINATION_PATH : Path for save results of predict
7. PREDICT_PRETRAINED_MODEL_PATH : Path of pre-trained model for predict

8. MIN_LEARNING_RATE : Min lr rate for cosine-lr scheduler
9. MAX_LEARNING_RATE : Max lr rate for cosine-lr scheduler, Basic lr rate for custom optimizer
10. MOMENTUM : Momeutum for sgd optimizer
11. BATCH_SIZE : Batch size scale for training
12. TEST_BATCH_SIZE : Batch size scale for evaluation
13. WIDTH : Model inference width
14. HEIGHT : Model inference height
15. LABEL_SMOOTHING_SCALE : Scale of soft label smoothing 
> ex) if 0.05, gt label of class 1 is [0.05, 0.95]
16. EPOCHS : Epochs of training
17. LOG_INTERVAL : Interval of training log print. (batch scale)
18. DIM : ConvMixer model channel dimmension
19. VISUALIZE_SAMPLE_NUM : Number of sample for grad-cam visualization once
20. VISUALIZE_PERIOD : Period of grad-cam visualization (batch scale)
21. VISUALIZE_LAYER : Model layer name for grad-cam visualization
22. PREDICT_CONFIDENCE_DIVIDE_RULE : Division rule of predict results by score
> ex) if [0.3, 0.6, 0.9], Predict results -> (0-30) OVER_0/, (30-60) OVER_30/, (60-90) OVER_60/, (90-100) OVER_90/ 
>   if no rule, []
23. ALE_LOSS_FOCAL_GAMMA_SCALE : Focal ALE loss gamma value 
24. EVALUATE_UNKNOWN_THRESHOLD : Unknown threshold value. if 0.3 and all output elements less than 0.3, predict result is unknown
25. PREDICT_UNKNOWN_THRESHOLD : Unknown threshold value for predict
26. PREDICT_UNCERTAIN_THRESHOLD : Uncertain score threshold value for predict. if 0.1, no-unknown and all output elements less than 0.1, save predict result of object at special result directory for uncertain object
27. TEACHER_WIDTH : Teacher model inference width for Knowledge Distillation
28. TEACHER_HEIGHT : Teacher model inference height for Knowledge Distillation
29. TEACHER_IS_COLOR : Teacher model inference channel. if False, gray or color
30. TEACHER_MODEL : Teacher model name

31. INIT_BEST_ACCURACY_FOR_PRETRAINED_MODEL : If training with pre-trained model, whether if use old best accuracy for save checkpoint
32. LABEL_SMOOTHING : If True, use label smoothing
33. IS_COLOR : Model inference channel. if True, color or gray
34. USE_RESNET18 : Use resnet-18 for main model 
35. USE_RESNEXT50 : Use resnext-50 for main model
36. VISUALIZE_GRAD_CAM : If True, visualize grad-cam result during training or visulization OFF
37. DEBUG_MODE : If True, display input tensor image (no train)
38. PREDICT_VERBOSE_SCORE : If True, append predict score at predict result of image file name 
39. KNOWLEDGE_DISTILLATION : If True, train with teacher model (Use Knowledge Distillation)
40. PREDICT_REMOVE_DIRECTORY_TREE : If True, The path of predict result is clean when execute predict.py or not clean
41. USE_CUSTOM_LR : If True, use custom lr-scheduler 
> https://github.com/inzapp/lr-scheduler
42. USE_ALE_LOSS : If True, use ALE loss 
> https://github.com/inzapp/absolute-logarithmic-error

*** 
Data Augmentation option (Albumentation)
43 ~ 52. ON(1), OFF(0)
