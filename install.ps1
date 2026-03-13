$files = @{
    "pose_landmarker_heavy.task" = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    "hand_landmarker.task" = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    "face_landmarker.task" = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
}

foreach ($file in $files.Keys) {
    Invoke-WebRequest $files[$file] -OutFile $file
}