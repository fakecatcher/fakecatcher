import cv2, time
from mtcnn.mtcnn import MTCNN

# load model
detector = MTCNN()

# initialize video source, default 0 (webcam)
video_path = 'videos/RollerCoasters.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output_mtcnn.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_count, tt = 0, 0

while cap.isOpened():
  ret, img = cap.read()
  if not ret:
    break

  frame_count += 1

  start_time = time.time()

  # prepare input
  result_img = img.copy()

  # inference, find faces
  detections = detector.detect_faces(result_img)

  # postprocessing
  for d in detections:
    x1, y1, w, h = d['box']
    keypoints = d['keypoints']

    x2 = x1 + w
    y2 = y1 + h

    # draw rects
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    # draw keypoints
    cv2.circle(result_img, (keypoints['left_eye']), 2, (0,155,255), 2, cv2.LINE_AA)
    cv2.circle(result_img, (keypoints['right_eye']), 2, (0,155,255), 2, cv2.LINE_AA)
    cv2.circle(result_img, (keypoints['nose']), 2, (0,155,255), 2, cv2.LINE_AA)
    cv2.circle(result_img, (keypoints['mouth_left']), 2, (0,155,255), 2, cv2.LINE_AA)
    cv2.circle(result_img, (keypoints['mouth_right']), 2, (0,155,255), 2, cv2.LINE_AA)

  # inference time
  tt += time.time() - start_time
  fps = frame_count / tt
  cv2.putText(result_img, 'FPS(mtcnn): %.2f' % (fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # visualize
  cv2.imshow('result', result_img)
  if cv2.waitKey(1) == ord('q'):
    break

  out.write(result_img)

cap.release()
out.release()
cv2.destroyAllWindows()
