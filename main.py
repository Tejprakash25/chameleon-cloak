import cv2
import torch
import numpy as np
import time
from torchvision import models, transforms
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights


device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Using device: {device}")


weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
model = models.segmentation.lraspp_mobilenet_v3_large(weights=weights)
model.eval().to(device)


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 120)),   
    transforms.ToTensor(),
])


def capture_background(cap, num_frames=60):
    print(f"⏳ Capturing background for {num_frames} frames... Please move out of the frame.")
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame.astype(np.float32))
        cv2.waitKey(30)
    avg_bg = np.mean(frames, axis=0).astype(np.uint8)
    print("✅ Background captured!")
    return avg_bg


mask_buffer = []
buffer_size = 5

def get_smoothed_mask(new_mask):
    mask_buffer.append(new_mask.astype(np.float32))
    if len(mask_buffer) > buffer_size:
        mask_buffer.pop(0)
    avg_mask = np.mean(mask_buffer, axis=0)
    smoothed = (avg_mask > 127).astype(np.uint8) * 255
    return smoothed


def get_mask(frame):
    inp = preprocess(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(inp)["out"][0]
    mask = output.argmax(0).byte().cpu().numpy()

   
    mask = np.where(mask == 15, 255, 0).astype(np.uint8)

   
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    
    mask = get_smoothed_mask(mask)
    return mask


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return

    background = capture_background(cap)

    print("⌛ Preparing cloak... step into the frame in 2 seconds!")
    time.sleep(2)

    print("🚀 Cloak effect running...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = get_mask(frame)

        
        inv_mask = cv2.bitwise_not(mask)

       
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        person_area = cv2.bitwise_and(frame, frame, mask=inv_mask)

       
        final = cv2.addWeighted(cloak_area, 1, person_area, 1, 0)

        cv2.imshow("🦎 Chameleon Cloak", final)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
