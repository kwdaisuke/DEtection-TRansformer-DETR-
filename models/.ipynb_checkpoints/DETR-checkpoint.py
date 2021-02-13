import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, Linear
import umbrella


class DETR(umbrella):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_queries = num_queries

        
    def deploy(self):      
        self.backbone = ResNet50()
        self.transformer = Transformer()
        
        self.model_dim = self.transformer.model_dim
        
        # Positional Encodings(object queries)
        self.encoder = PositionalEmbeddingSine(pos_features=self.model_dim//2, normalize = True)
        self.class_embed = Linear(num_classes + 1, )
        self.embedding = Linear(self.model_dim)
        self.activation = ReLU()
        
        # Propagate
        self.call()
        
        
    def call(self):    
        x = self.backbone(x)
        x = self.downsample_masks(masks, x)
        x = self.encoder(x)
        x = self.transformer(self.input_proj(x))
        
        outputs_class = self.class_embeded(x)
        
        x = self.activation(self.embedding(x))
        x = self.activation(self.embedding(x))
        outputs_coordinate = tf.sigmoid(self.embedding(x))
        
        output = {"pred_logits": outputs_class[-1],
                  "pred_boxes": outputs_coordinate[-1]}
        
        if post_proces:
            output = self.post_process(output)
        return output
    
    
    def downsample_masks(self, masks, x):
        x = tf.cast(masks, tf.int32)
        x = tf.expand_dims(x, -1)
        x = tf.compat.v1.image_resize_nearest_neighbor(
                x, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False)
        x = tf.squeeze(x, -1)
        x = tf.cast(x, tf.bool)
        return x
   

    def post_process(self, output):
        logits, boxes = [output[k] for k in ["pred_logits", ,"pred_boxes"]]
        probs = tf.nn.softmax(logits, axis=-1)[..., :-1]
        scores = tf.reduce_max(probs, axis=-1)
        labels = tf.argmax(probs, axis=-1)
        boxes = frame(boxes)
        
        output = {"scores": scores,
                   "labels": labels.
                   "boxes": boxes}
        return output
                    
    
    