class ModelComplex():
    def __init__(self, yolo, mono):
        from mono import MONO 
        self.mono = MONO(mono) 
        from ultralytics import YOLO
        self.yolo = YOLO(yolo)
        
    def __call__(self, images, conf=0.5):
        segments = self.yolo(images, conf=conf, device='cpu')
        depths, instance_depths = self.mono(images, segments) 
        return [(x.plot(), y, z) for x, y, z in zip(segments, depths, instance_depths)]
        