import cv2
import numpy as np
import mlx.core as mx

from model.model import resnet20



model = resnet20()

model.load_weights("savemodel/model.npz")
mx.eval(model.parameters())

h = 240
w = 240
if __name__ == "__main__":
    cap = cv2.VideoCapture("data/video/IMG_4290.MOV")
    # fourcc = cv2.VideoWriter_fourcc(*'PIMI')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
    while (True):
        # 擷取影像
        ret, frame = cap.read()
        if ret is not True:
            print("Can't receive frame (stream end?). Exiting ...")
            exit()
        raw = frame
        frame = cv2.resize(frame, (w, h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame/255
        frame = frame.astype(np.float32)
        

        depth_map = model(mx.array(frame))

        depth_map = np.array(depth_map).squeeze()

        depth_map = depth_map*256  
        depth_map = depth_map.astype(np.uint8)
       
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_HSV)

        # depth_map = cv2.resize(depth_map, (640, 360))
        raw = cv2.resize(raw, (w, h))

        Hori = np.concatenate((raw, depth_map), axis=1) 

        # out.write(Hori)
        cv2.imshow('result', Hori)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()