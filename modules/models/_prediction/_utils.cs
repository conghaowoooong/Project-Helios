/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 19:17:12
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-05 23:34:42
 * @Description: file content
 */

using System;
using System.Collections.Generic;
using NumSharp;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static modules.models.helpMethods.HelpMethods;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using modules.models.Base;

namespace modules.models.Prediction
{
    public static class Loss
    {
        ///FUNCTION_NAME: ADE
        ///<summary>
        ///        Calculate `ADE` or `minADE` by `tensorflow`.
        ///        
        ///            Return `ADE` when input_shape = [batch, pred_frames, 2];
        ///</summary>
        ///<param name="pred"> pred traj, shape = `[batch, pred, 2]` </param>
        ///<param name="GT"> ground truth future traj, shape = `[batch, pred, 2]` </param>
        ///<return name="loss_ade">: </return>
        public static Tensor ADE(Tensor pred, Tensor GT)
        {
            if (len(pred.shape) == 4)
            {
                var all_ade = tf.reduce_mean(tf_norm(pred - tf.expand_dims(GT, axis: 1), ord: 2, axis: -1), axis: -1);
                var best_ade = tf.reduce_min(all_ade, axis: new int[] { 1 });
                return tf.reduce_mean(best_ade);
            }
            else if (len(pred.shape) == 3)
            {
                return tf.reduce_mean(tf_norm(pred - GT, ord: 2, axis: 2));
            }
            else
            {
                return null;
            }
        }

        ///FUNCTION_NAME: FDE
        ///<summary>
        ///        Calculate `FDE` or `minFDE` by `tensorflow`.
        ///        
        ///            Return `FDE` when input_shape = [batch, pred_frames, 2];
        ///</summary>
        ///<param name="pred"> pred traj, shape = `[batch, pred, 2]` </param>
        ///<param name="GT"> ground truth future traj, shape = `[batch, pred, 2]` </param>
        ///<return name="loss_ade"> </return>
        public static Tensor FDE(Tensor pred, Tensor GT)
        {
            if (len(pred.shape) == 4) // [batch, K, pred, 2]
            {
                var all_ade = tf.reduce_mean(tf_norm(pred - tf.expand_dims(GT, axis: 1), ord: 2, axis: -1), axis: -1);
                var best_ade_index = tf.arg_min(all_ade, 1);

                var pred_best = tf.gather(
                    pred,
                    tf.transpose(tf.stack((tf.range(0, pred.shape[0], dtype: tf.int32), best_ade_index)))
                );
                return tf.reduce_mean(
                    tf_norm(tf.transpose(pred_best - GT, (1, 0, 2))[-1], ord: 2, axis: 1)
                );
            }
            else if (len(pred.shape) == 3) // [batch, pred, 2]
            {
                return tf.reduce_mean(tf_norm(tf.transpose(pred - GT, (1, 0, 2))[-1], ord: 2, axis: 1));
            }
            else
            {
                return null;
            }
        }
    }

    public static class Process
    {
        // TODO
    }

    public static class Utils
    {
        public static List<TrainAgentManager> load_dataset_files(TrainArgs args, string dataset)
        {
            dir_check("./dataset_npz");
            var dm = new TrainDataManager(args, prepare_type: "noprepare");
            var agents = dm.prepare_train_files(new List<DatasetManager> { new DatasetManager(args, dataset) });
            return agents;
        }

        ///FUNCTION_NAME: getInputs_onlyTraj
        ///<summary>
        ///    Get inputs for models who only takes `obs_traj` as input.
        ///
        ///</summary>
        ///<param name="input_agents"> a list of input agents, type = `List[Agent]` </param>
        ///<return name="model_inputs"> a list of traj tensor, `len(model_inputs) = 1` </return>
        public static (List<Tensor> model_inputs, Tensor gt) getInputs_onlyTraj(List<TrainAgentManager> input_agents)
        {
            List<List<Tensor>> model_inputs = new List<List<Tensor>>();
            List<Tensor> gt = new List<Tensor>();

            model_inputs.append(new List<Tensor>());
            foreach (var agent in input_agents)
            {
                model_inputs[0].append(agent.traj);
                gt.append(agent.groundtruth);
            }

            List<Tensor> tensor_inputs = new List<Tensor>();
            foreach (var index in range(len(model_inputs)))
            {
                tensor_inputs.append(tf.cast(tf.stack(model_inputs[index].ToArray()), tf.float32));
            }

            var gt_input = tf.cast(tf.stack(gt.ToArray()), tf.float32);
            return (tensor_inputs, gt_input);
        }

        ///FUNCTION_NAME: getForwardDataset_onlyTraj
        ///<summary>
        ///    Get inputs for models who only takes `obs_traj` as input.
        ///
        ///</summary>
        ///<param name="input_agents"> a list of input agents, type = `List[Agent]` </param>
        public static IDatasetV2 getForwardDataset_onlyTraj(List<TrainAgentManager> input_agents)
        {
            var trajs = new List<NDArray>();
            foreach (var agent in input_agents)
            {
                trajs.append(agent.traj);
            }
            return tf.data.Dataset.from_tensor_slices(new Tensor(np.stack(trajs.ToArray()), tf.float32));
        }
    }
}