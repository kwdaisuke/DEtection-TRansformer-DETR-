import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU
from model import Umbrella
from backbone import ResNet50
from position_embeddings import PositionEmbeddingSine
from transformer import Transformer
from model import Linear

class DETR(Umbrella):
    def __init__(self,  *args, **kwargs):
        self.num_classes = 91
        super().__init__(*args, **kwargs)
        #elf.num_queries = num_queries

        
    def deploy(self):      
        self.backbone = ResNet50()
        self.transformer = Transformer()
        
        self.model_dim = self.transformer.model_dim
        
        # Positional Encodings(object queries)
        self.encoder = PositionEmbeddingSine(num_pos_features=self.model_dim//2, normalize = True)
        self.class_embed = Linear(self.num_classes + 1, )
        self.embedding = Linear(self.model_dim)
        self.activation = ReLU()
        
        # Propagate
        self.call()
        
        
    def call(self):    
        self.x = self.backbone(self.x)
        self.x = self.downsample_masks(masks, self.x)
        self.x = self.encoder(self.x)
        self.x = self.transformer(self.input_proj(self.x))
        
        outputs_class = self.class_embeded(self.x)
        
        self.x = self.activation(self.embedding(self.x))
        self.x = self.activation(self.embedding(self.x))
        outputs_coordinate = tf.sigmoid(self.embedding(self.x))
        
        output = {"pred_logits": outputs_class[-1],
                  "pred_boxes": outputs_coordinate[-1]}
        
        if post_proces:
            output = self.post_process(output)
        return output
    
    
    def downsample_masks(self, masks, x):
        self.x = tf.cast(masks, tf.int32)
        self.x = tf.expand_dims(self.x, -1)
        self.x = tf.compat.v1.image_resize_nearest_neighbor(
                self.x, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False)
        self.x = tf.squeeze(self.x, -1)
        self.x = tf.cast(self.x, tf.bool)
        return x
   

    def post_process(self, output):
        logits, boxes = [output[k] for k in ["pred_logits" ,"pred_boxes"]]
        probs = tf.nn.softmax(logits, axis=-1)[..., :-1]
        scores = tf.reduce_max(probs, axis=-1)
        labels = tf.argmax(probs, axis=-1)
        boxes = frame(boxes)
        
        output = {"scores": scores,
                   "labels": labels,
                   "boxes": boxes}
        return output
                    
    
    
if __name__ == "__main__":
    detr = DETR()