/*
 * @Author: Conghao Wong
 * @Date: 2021-01-28 23:59:46
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-29 02:01:21
 * @Description: file content
 */

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using static models.Prediction.Utils;

using static models.HelpMethods;
using models.Managers.ArgManagers;
using models.Managers.AgentManagers;
using models.Managers.TrainManagers;
using models.Managers.DatasetManagers;

namespace models.Prediction
{
    class Linear : BasePredictionModel
    {
        private NDArray x_obs;
        private Tensor x;
        private Tensor x_p;

        private Tensor A_p;
        private Tensor W;

        public Linear(TrainArgsManager args) : base(args)
        {
            this.x_obs = np.arange(this.args.obs_frames) / (this.args.obs_frames - 1);
        }

        public override Model load_from_checkpoint(string model_path)
        {
            this.create_model();
            return this.model;
        }

        public override Tensors mid_process(Tensors model_inputs, Dictionary<string, object> kwargs = null)
        {
            return this.linear_predict_batch(model_inputs);
        }

        List<Tensor> linear_predict_batch(Tensors inputs)
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
            var list_results = new List<Tensor>();
            list_results.append(results);
            return list_results;
        }

        public void create_model(string flag = "linear")
        {
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

            this.x = tf.range(0, this.args.obs_frames, dtype: tf.float32);
            this.x_p = tf.range(this.args.obs_frames, this.args.obs_frames + this.args.pred_frames, dtype: tf.float32);
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
    }
}