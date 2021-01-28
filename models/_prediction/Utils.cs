/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 19:17:12
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-29 01:02:38
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
using static models.HelpMethods;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using models.Managers.TrainManagers;
using models.Managers.AgentManagers;
using models.Managers.ArgManagers;

namespace models.Prediction{
    public static class Utils{
        public static List<TrainAgentManager> load_dataset_files(TrainArgsManager args, string dataset){
            dir_check("./dataset_npz");
            var agents_save_format = "./dataset_npz/{0}/agent.dat";
            List<TrainAgentManager> agents;
            if (!file_exist(String.Format(agents_save_format, dataset))){
                var dm = new TrainDataManager(args, prepare_type:"noprepare");
                agents = dm.prepare_train_files(new List<Managers.TrainManagers.DatasetManager> {new models.Managers.TrainManagers.DatasetManager(args, dataset)});
            } else {
                agents = read_file<List<TrainAgentManager>>(String.Format(agents_save_format, dataset));
            }
            return agents;
        }
        
        public static (List<Tensor> model_inputs, Tensor gt) getInputs_onlyTraj(List<TrainAgentManager> input_agents){
            List<List<Tensor>> model_inputs = new List<List<Tensor>>();
            List<Tensor> gt = new List<Tensor>();

            model_inputs.append(new List<Tensor>());
            foreach (var agent in input_agents){
                model_inputs[0].append(agent.get_traj());
                gt.append(agent.get_future_traj());
            }

            List<Tensor> tensor_inputs = new List<Tensor>();
            foreach (var index in range(len(model_inputs))){
                tensor_inputs.append(tf.cast(tf.stack(model_inputs[index].ToArray()), tf.float32));
            }

            var gt_input = tf.cast(tf.stack(gt.ToArray()), tf.float32);
            return (tensor_inputs, gt_input);
        }

        public static IDatasetV2 getForwardDataset_onlyTraj(List<TrainAgentManager> input_agents){
            var trajs = new List<NDArray>();
            foreach (var agent in input_agents){
                trajs.append(agent.get_traj());
            }
            return tf.data.Dataset.from_tensor_slices(new Tensor(np.stack(trajs.ToArray()), tf.float32));
        }

    }


    public static class Loss{
        public static Tensor ADE(Tensor pred, Tensor GT){
            if (len(pred.shape) == 4){
                var all_ade = tf.reduce_mean(tf_norm(pred - tf.expand_dims(GT, axis:1), ord:2, axis:-1), axis:-1);
                var best_ade = tf.reduce_min(all_ade, axis:new int[] {1});
                return tf.reduce_mean(best_ade);
            } else if (len(pred.shape) == 3){
                return tf.reduce_mean(tf_norm(pred - GT, ord:2, axis:2));
            } else {
                return null;
            }
        }

        public static Tensor FDE(Tensor pred, Tensor GT){
            if (len(pred.shape) == 4) // [batch, K, pred, 2]
            {
                var all_ade = tf.reduce_mean(tf_norm(pred - tf.expand_dims(GT, axis:1), ord:2, axis:-1), axis:-1);
                var best_ade_index = tf.arg_min(all_ade, 1);

                var pred_best = tf.gather(
                    pred,
                    tf.transpose(tf.stack((tf.range(0, pred.shape[0], dtype:tf.int32), best_ade_index)))
                );
                return tf.reduce_mean(
                    tf_norm(tf.transpose(pred_best - GT, (1, 0, 2))[-1], ord:2, axis:1)
                );
            }
            else if (len(pred.shape) == 3) // [batch, pred, 2]
            {
                return tf.reduce_mean(tf_norm(tf.transpose(pred - GT, (1, 0, 2))[-1], ord:2, axis:1));
            } else {
                return null;
            }
        }
    }
}