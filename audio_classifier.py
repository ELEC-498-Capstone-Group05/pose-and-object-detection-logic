"""
YAMNet Audio Detection for Baby Monitor
Live audio monitoring with real-time sound classification
"""

import numpy as np
import sounddevice as sd
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow.lite as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

import csv
import time
import threading
from collections import deque
import logging

logger = logging.getLogger(__name__)

class AudioClassifier:
    def __init__(self, model_path, labels_path, callback=None, device_index=None, 
                 threshold=0.3, sample_rate=16000, window_size=0.975, hop_size=0.5, gain=5.0):
        self.model_path = model_path
        self.labels_path = labels_path
        self.callback = callback
        self.device_index = device_index
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.gain = gain
        
        self.buffer_len = int(sample_rate * window_size)
        self.audio_buffer = deque(maxlen=self.buffer_len)
        self.buffer_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.stream = None
        self.interpreter = None
        self.class_names = self._load_class_names()
        
        # Load model
        try:
            logger.info(f"Loading audio model from {model_path}")
            self.interpreter = Interpreter(model_path=str(model_path))
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info("Audio model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            raise

        self.latest_top_label = "Listening..."
        self.latest_db = -100.0

    def _load_class_names(self):
        class_names = {}
        try:
            with open(self.labels_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        class_names[int(row[0])] = row[2]
        except Exception as e:
            logger.error(f"Error loading audio labels: {e}")
        return class_names

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Calculate dB
        rms = np.sqrt(np.mean(indata**2))
        db = 20 * np.log10(rms) if rms > 1e-9 else -100
        self.latest_db = float(max(-100, db))

        with self.buffer_lock:
            self.audio_buffer.extend(indata[:, 0])

    def _inference_loop(self):
        logger.info("Starting audio inference loop")
        while self.running:
            start_time = time.time()
            
            # Get audio window
            audio_data = None
            with self.buffer_lock:
                if len(self.audio_buffer) >= self.buffer_len:
                    audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
            
            if audio_data is not None:
                try:
                    # Apply gain and clip
                    audio_data = audio_data * self.gain
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    
                    # Inference
                    detections = self._run_inference(audio_data)
                    
                    if detections:
                        # Get highest scoring class
                        top_id = sorted(detections.items(), key=lambda x: x[1]['score'], reverse=True)[0][0]
                        self.latest_top_label = detections[top_id]['name']
                    else:
                        self.latest_top_label = "Background"

                    if self.callback and detections:
                        self.callback(detections)
                except Exception as e:
                    logger.error(f"Error in audio inference: {e}")
            
            # Sleep remainder of hop size
            elapsed = time.time() - start_time
            sleep_time = max(0.1, self.hop_size - elapsed)
            time.sleep(sleep_time)

    def _run_inference(self, audio_data):
        # Prepare input
        input_shape = self.input_details[0]['shape']
        expected_length = input_shape[-1] if len(input_shape) > 0 else 15600
        
        if len(audio_data) != expected_length:
             if len(audio_data) < expected_length:
                 audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)))
             else:
                 audio_data = audio_data[:expected_length]
        
        # Add batch dimension if needed
        if len(self.input_details[0]['shape']) > 1 and len(audio_data.shape) == 1:
            input_tensor = np.expand_dims(audio_data, axis=0)
        else:
            input_tensor = audio_data

        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        avg_scores = np.mean(scores, axis=0)
        
        results = {}
        # Get top 5
        top_indices = np.argsort(avg_scores)[-5:][::-1]
        for idx in top_indices:
            score = float(avg_scores[idx])
            if score > self.threshold:
                name = self.class_names.get(idx, f'Class {idx}')
                results[idx] = {'score': score, 'name': name}
                
        return results

    def _find_working_device(self):
        """Scan for a working input device if preferred one fails or is not specified"""
        try:
            devices = sd.query_devices()
            # If specific device requested and valid
            if self.device_index is not None:
                if self.device_index < len(devices) and devices[self.device_index]['max_input_channels'] > 0:
                    return self.device_index
                logger.warning(f"Requested device {self.device_index} invalid or has no input. Scanning...")

            # Try to find a USB device first (common for webcams/mics)
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    if 'usb' in dev['name'].lower():
                        return i
            
            # Fallback to default
            default_in = sd.default.device[0]
            if default_in < len(devices) and devices[default_in]['max_input_channels'] > 0:
                return default_in
            
            # Any input device
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    return i
                    
        except Exception as e:
            logger.error(f"Error enumerating audio devices: {e}")
        
        return None

    def start(self):
        if self.running:
            return
            
        # Select device
        device_id = self._find_working_device()
        if device_id is None:
            logger.error("No suitable audio input device found")
            return

        device_name = sd.query_devices(device_id)['name']
        logger.info(f"Starting audio stream on device [{device_id}] {device_name}")
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.1),
                device=device_id
            )
            self.stream.start()
            self.running = True
            
            self.thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.running = False
            return False


    def stop(self):
        logger.info("Stopping audio classifier")
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
        if self.thread:
            self.thread.join(timeout=1.0)
