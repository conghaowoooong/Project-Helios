/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 19:17:12
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-07 09:55:52
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
                //var best_ade = tf.reduce_min(all_ade, axis: new int[] { 1 });   //BUG: tf.reduce_min is tf1 version
                var best_ade = tf.cast(np.min(all_ade.numpy(), 1), tf.float32);
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
                // var best_ade_index = tf.arg_min(all_ade, 1);    // BUG: tf.arg_min is tf1 version
                var best_ade_index = tf.cast(np.argmin(all_ade.numpy(), 1), tf.int32);
                
                var pred_best_ = new List<Tensor>();
                foreach (var index in range(best_ade_index.shape[0])){
                    pred_best_.append(pred[index][best_ade_index[index].numpy()]);
                }
                var pred_best = tf.stack(pred_best_);
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
        ///FUNCTION_NAME: move
        ///<summary>
        ///        Move trajectories to (0, 0) according to the reference point.
        ///        Default reference point is the last obsetvation.
        ///
        ///</summary>
        ///<param name="trajs"> observations, shape = `[(batch,) obs, 2]` </param>
        ///<param name="ref"> reference point, default is `-1` </param>
        ///<return name="traj_moved"> moved trajectories </return>
        public static (Tensor, Dictionary<string, Tensor>) move(Tensor trajs, int reff=-1){
            Tensor ref_point;
            if (len(trajs.shape) == 3) {
                ref_point = tf.split(trajs, trajs.shape[1], 1)[reff < 0 ? trajs.shape[1]+reff : reff];
            } else {
                ref_point = tf.split(trajs, trajs.shape[0], 0)[reff < 0 ? trajs.shape[0]+reff : reff];
            }   // shape is [batch, 1, 2] or [1, 2]


            var traj_moved = trajs - ref_point;
            var para_dict = new Dictionary<string, Tensor>();
            para_dict.Add("ref_point", ref_point);
            return (traj_moved, para_dict);
        }

        ///FUNCTION_NAME: move_back
        ///<summary>
        ///        Move trajectories back to their original positions.
        ///
        ///</summary>
        ///<param name="trajs"> trajectories moved to (0, 0) with reference point, shape = `[(batch,) (K,) pred, 2]` </param>
        ///<param name="para_dict"> a dict of used parameters, `ref_point:tf.Tensor` </param>
        public static Tensor move_back(Tensor trajs, Dictionary<string, Tensor> para_dict){
            var ref_point = para_dict["ref_point"]; // shape = [(batch,) 1, 2]

            Tensor traj_moved;
            if (len(ref_point.shape) == len(trajs.shape)){
                traj_moved = trajs + ref_point;
            } else {    // [(batch,) K, pred, 2]
                traj_moved = trajs + tf.expand_dims(ref_point, -3);
            }
            return traj_moved;
        }

        ///FUNCTION_NAME: rotate
        ///<summary>
        ///        Rotate trajectories to referce angle.
        ///        
        ///</summary>
        ///<param name="trajs"> observations, shape = `[(batch,) obs, 2]` </param>
        ///<param name="ref"> reference angle, default is `0` </param>
        ///<return name="traj_rotated"> moved trajectories </return>
        public static (Tensor, Dictionary<string, Tensor>) rotate(Tensor trasj, int reff=0){
            // TODO
            return (null, null);
        }

        ///FUNCTION_NAME: rotate_back
        ///<summary>
        ///        Rotate trajectories back to their original angles.
        ///
        ///</summary>
        ///<param name="trajs"> trajectories, shape = `[(batch, ) pred, 2]` </param>
        ///<param name="para_dict"> a dict of used parameters, `rotate_matrix:tf.Tensor` </param>
        public static Tensor rotate_back(Tensor trajs, Dictionary<string, Tensor> para_dict){
            // TODO
            return null;
        }

        ///FUNCTION_NAME: scale
        ///<summary>
        ///        Scale trajectories' direction vector into (x, y), where |x| <= 1, |y| <= 1.
        ///        Reference point when scale is the `last` observation point.
        ///
        ///</summary>
        ///<param name="trajs"> input trajectories, shape = `[(batch,) obs, 2]` </param>
        ///<param name="ref"> reference length, default is `1` </param>
        ///<return name="traj_scaled"> scaled trajectories </return>
        public static (Tensor, Dictionary<string, Tensor>) scale(Tensor trasj, float reff=1.0f){
            // TODO
            return (null, null);
        }

        ///FUNCTION_NAME: scale_back
        ///<summary>
        ///        Scale trajectories back to their original.
        ///        Reference point is the `first` prediction point.
        ///
        ///</summary>
        ///<param name="trajs"> trajectories, shape = `[(batch,) (K,) pred, 2]` </param>
        ///<param name="para_dict"> a dict of used parameters, contains `scale:tf.Tensor` </param>
        public static Tensor scale_back(Tensor trajs, Dictionary<string, Tensor> para_dict){
            // TODO
            return null;
        }

        public static Tensors update(Tensors neww, Tensors old){
            if (len(neww) < len(old)){
                foreach (int index in range(len(neww), len(old))){
                    neww.Add(old[index]);
                }
                return neww;
            } else {
                return neww;
            }
        }
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

        ///FUNCTION_NAME: getInputs_TrajWithMap
        ///<summary>
        ///    Get inputs for models who takes `obs_traj` and one type of map `map, map_para` as input.
        ///
        ///</summary>
        ///<param name="input_agents"> a list of input agents, type = `List[Agent]` </param>
        ///<return name="model_inputs"> a list of traj tensor, `len(model_inputs) = 3` </return>
        public static (List<Tensor> model_inputs, Tensor gt) getInputs_TrajWithMap(List<TrainAgentManager> input_agents)
        {
            List<Tensor> traj_inputs = new List<Tensor>();
            List<Tensor> map_inputs = new List<Tensor>();
            List<Tensor> para_inputs = new List<Tensor>();
            List<Tensor> gt = new List<Tensor>();

            foreach (var agent in input_agents)
            {
                traj_inputs.append(agent.traj);
                map_inputs.append(agent.fusionMap);
                para_inputs.append(agent.real2grid);
                gt.append(agent.groundtruth);
            }

            var traj_inputs_ = tf.cast(tf.stack(traj_inputs.ToArray()), tf.float32);
            var map_inputs_ = tf.cast(tf.stack(map_inputs.ToArray()), tf.float32);
            var para_inputs_ = tf.cast(tf.stack(para_inputs.ToArray()), tf.float32);
            var gt_input_ = tf.cast(tf.stack(gt.ToArray()), tf.float32);
            return (new List<Tensor> {traj_inputs_, map_inputs_, para_inputs_}, gt_input_);
        }

        ///FUNCTION_NAME: getForwardDataset_onlyTraj
        ///<summary>
        ///    Get inputs for models who only takes `obs_traj` as input.
        ///
        ///</summary>
        ///<param name="input_agents"> a list of input agents, type = `List[Agent]` </param>
        public static Tensors getForwardDataset_onlyTraj(List<TrainAgentManager> input_agents)
        {
            var trajs = new List<NDArray>();
            foreach (var agent in input_agents)
            {
                trajs.append(agent.traj);
            }
            return new Tensor(np.stack(trajs.ToArray()), tf.float32);
        }

        public static Tensors getForwardDataset_TrajWithMap(List<TrainAgentManager> input_agents)
        {
            var trajs = new List<NDArray>();
            var maps = new List<NDArray>();
            var paras = new List<NDArray>();
            foreach (var agent in input_agents)
            {
                trajs.append(agent.traj);
                maps.append(agent.fusionMap);
                paras.append(agent.real2grid);
            }

            var trajs_ = tf.cast(tf.stack(trajs.ToArray()), tf.float32);
            var maps_ = tf.cast(tf.stack(maps.ToArray()), tf.float32);
            var paras_ = tf.cast(tf.stack(paras.ToArray()), tf.float32);
           
            return new Tensors {trajs_, maps_, paras_};
        }
    }
}