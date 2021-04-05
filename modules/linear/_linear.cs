/*
 * @Author: Conghao Wong
 * @Date: 2021-03-31 15:51:46
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-05 22:33:19
 * @Description: file content
 */


using NumSharp;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using static modules.models.helpMethods.HelpMethods;

using Prediction = modules.models.Prediction;

namespace modules.Linear
{
    class LinearModel : Prediction.Model {
        Tensor x;
        Tensor x_p;
        Tensor A_p;
        Tensor W;

        public LinearModel(
            Prediction.TrainArgs args,
            Prediction.Structure training_structure = null
        ) : base(args, training_structure){
            
            Tensor P;
            var diff_weights = this.args.diff_weights;
            if (diff_weights == 0)
            {
                P = tf.diag(tf.ones(this.args.obs_frames));
            }
            else
            {
                P = tf.diag(tf.nn.softmax(
                    tf.pow(tf.cast(tf.range(1, 1 + this.args.obs_frames), tf.float32), diff_weights)
                ));
            }

            this.x = tf.cast(np.arange(this.args.obs_frames), tf.float32);
            this.x_p = tf.cast(np.arange(this.args.pred_frames) + this.args.obs_frames, tf.float32);
            var A = tf.transpose(tf.stack(new Tensor[] {
                tf.ones((this.args.obs_frames), dtype:tf.float32),
                this.x
            }));
            this.A_p = tf.transpose(tf.stack(new Tensor[] {
                tf.ones((this.args.pred_frames), dtype:tf.float32),
                this.x_p
            }));
            this.W = tf.matmul(tf.matmul(ndarray_inv((tf.matmul(tf.matmul(tf.transpose(A), P), A)).numpy()).astype(np.float32), tf.transpose(A)), P);            
        }

        public override Tensors call(Tensors inputs, bool training = false, dynamic mask = null)
        {
            var input = tf.transpose(inputs[0], (2, 0, 1));

            var x = tf.expand_dims(input[0], axis: -1);
            var y = tf.expand_dims(input[1], axis: -1);

            var Bx = tf_batch_matmul(this.W, x);
            var By = tf_batch_matmul(this.W, y);

            var results = tf.stack(new Tensor[] {
                tf_batch_matmul(this.A_p, Bx),
                tf_batch_matmul(this.A_p, By)
            });

            results = tf.transpose(results, (3, 1, 2, 0))[0];
            return results;
        }
    }


    class Linear : Prediction.Structure {
        NDArray x_obs;
        
        public Linear(Prediction.TrainArgs args) : base(args){
            args.load = "linear";
            this.x_obs = np.arange(this.args.obs_frames) / (this.args.obs_frames - 1);
        }

        public override Prediction.TrainArgs load_args()
        {
            return this.args;
        }

        public override Prediction.Model load_from_checkpoint(string model_path)
        {
            (this.model, _) = this.create_model();
            return this.model;
        }

        public override (Prediction.Model, OptimizerV2) create_model()
        {
            var optimizer = keras.optimizers.Adam(this.args.lr);
            return (new LinearModel(this.args), optimizer);
        }
    }
}