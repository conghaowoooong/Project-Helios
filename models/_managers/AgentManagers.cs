/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 20:04:30
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-03-26 15:56:11
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

namespace models.Managers.AgentManagers
{
    [Serializable]
    public class BaseAgentManager
    {
        Array _traj;
        Array _traj_future;
        Array _traj_pred;
        Array _traj_pred_linear;
        Array _traj_map;
        Array _social_map;
        Array _real2grid;
        Array _frame_list;
        Array _frame_list_future;

        (float ade, float fde) _loss;

        float _version_ = 2.0f;

        public void copy()
        {
            // TODO: copy() in BaseAgentManager
        }


        // 1. Historical Trajectory
        public NDArray traj{
            get {
                return np.array(this._traj).astype(np.float32);
            }
            set {
                this._traj = value.astype(np.float32).ToMuliDimArray<float>();
            }
        }

        // 2. Prediction Trajectory
        public NDArray pred{
            get {
                return np.array(this._traj_pred).astype(np.float32);
            }
            set {
                this._traj_pred = value.astype(np.float32).ToMuliDimArray<float>();
            }
        }

        // Frame List
        public NDArray frame_list {
            get {
                return np.concatenate(new NDArray[] {
                    np.array(this._frame_list),
                    np.array(this._frame_list_future)
                });
            }
            set {
                this._frame_list = value.ToArray<int>();
            }
        }

        // Future Frame List
        public NDArray frame_list_future {
            get {
                return this._frame_list_future;
            }
            set {
                this._frame_list_future = value.ToArray<int>();
            }
        }

        // 3. Linear Prediction
        public NDArray pred_linear {
            get {
                return np.array(this._traj_pred_linear).astype(np.float32);
            }
            set {
                this._traj_pred_linear = value.astype(np.float32).ToMuliDimArray<float>();
            }
        }

        // 4. Future Ground Truth
        public NDArray groundtruth {
            get {
                return np.array(this._traj_future).astype(np.float32);
            }
            set {
                this._traj_future = value.astype(np.float32).ToMuliDimArray<float>();
            }
        }

        // 5. Trajectory Map
        public NDArray trajMap {
            get {
                return np.array(this._traj_map).astype(np.float32);
            }
        }

        public MapManager trajMapManager{
            set {
                var full_map = value.guidance_map;
                var half_size = value.args.map_half_size;
                
                var center_pos = value.real2grid(this.traj["-1, :"]);
                var original_map = full_map[String.Format("{0}, {1}",
                    models.HelpMethods.get_slice_index(np.maximum(center_pos[0] - half_size, 0), np.minimum(center_pos[0] + half_size, full_map.shape[0])),
                    models.HelpMethods.get_slice_index(np.maximum(center_pos[1] - half_size, 0), np.minimum(center_pos[1] + half_size, full_map.shape[1]))
                )];
                
                var final_map = tf.image.resize(tf.expand_dims(original_map, axis: -1), (2 * half_size, 2 * half_size));
                this._traj_map = final_map.numpy()[":, :, 0"].astype(np.float32).ToMuliDimArray<float>();
                this._real2grid = value.real2grid_paras().astype(np.float32).ToMuliDimArray<float>();
            }
        }

        // 6. Social Map
        public NDArray socialMap {
            get {
                return np.array(this._social_map).astype(np.float32);
            }
        }

        public MapManager socialMapManager {
            set {
                var half_size = value.args.map_half_size;
                var center_pos = value.real2grid(this.traj["-1, :"]);
                var full_map = value.full_map;

                var original_map = full_map[String.Format("{0}, {1}",
                    models.HelpMethods.get_slice_index(np.maximum(center_pos[0] - half_size, 0), np.minimum(center_pos[0] + half_size, full_map.shape[0])),
                    models.HelpMethods.get_slice_index(np.maximum(center_pos[1] - half_size, 0), np.minimum(center_pos[1] + half_size, full_map.shape[1]))
                )];
                var final_map = tf.image.resize(tf.expand_dims(original_map, axis: -1), (2 * half_size, 2 * half_size));

                this._social_map = final_map.numpy()[":, :, 0"].astype(np.float32).ToMuliDimArray<float>();
                this._real2grid = value.real2grid_paras().astype(np.float32).ToMuliDimArray<float>();
            }
        }

        // 7. Fusion Map
        public NDArray fusionMap {
            get {
                if (this._social_map == null)
                {
                    return 0.5 * this.trajMap;
                }
                else
                {
                    return 0.5 * this.socialMap + 0.5 * this.trajMap;
                }
            }
        }

        // 8. Loss
        public (float ade, float fde) loss {
            get {
                return this._loss;
            }
            set {
                this._loss = value;
            }
        }

        public NDArray rotate_map(NDArray map, float rotate_angle){
            // TODO rotate map in BaseAgentManager
            return np.array((0));
        }
    }


    [Serializable]
    public class TrainAgentManager : BaseAgentManager
    {
        List<Array> _neighbor_traj;
        List<Array> _neighbor_traj_linear_pred;
        bool linear_predict;
        public int obs_length;
        public int total_frame;
        public int neighbor_number;
        

        public TrainAgentManager(
            EntireTrajectory target_agent,
            List<EntireTrajectory> neighbor_agents,
            NDArray frame_list,
            int start_frame,
            int obs_frame,
            int end_frame,
            int frame_step = 1,
            bool add_noise = false,
            bool linear_predict = true
        )
        {
            this.linear_predict = linear_predict;

            // trajectory info
            this.obs_length = (obs_frame - start_frame) / frame_step;
            this.total_frame = (end_frame - start_frame) / frame_step;

            // Trajectory
            var whole_traj = target_agent.traj[String.Format("{0}:{1}:{2}", start_frame, end_frame, frame_step)].copy();
            var frame_list_current = np.array(frame_list)[String.Format("{0}:{1}:{2}", start_frame, end_frame, frame_step)].copy();

            if (add_noise)
            {
                // TODO: add noise in TrainAgentManager
            }

            var index1 = String.Format(":{0}", this.obs_length);
            var index2 = String.Format("{0}:", this.obs_length);
            this.frame_list = frame_list_current[index1];
            this.traj = whole_traj[index1];
            this.groundtruth = whole_traj[index2];
            this.frame_list_future = frame_list_current[index2];

            if (linear_predict)
            {
                this.pred_linear = predict_linear_for_person(
                    this.traj,
                    time_pred: this.total_frame
                )[index2];
            }

            // neighbor info
            var neighbor_traj = new List<NDArray>();
            var neighbor_traj_linear_pred = new List<NDArray>();
            foreach (var neighbor in neighbor_agents)
            {
                var current_neighbor_traj = neighbor.traj[String.Format("{0}:{1}:{2}", start_frame, end_frame, frame_step)].copy();
                if (current_neighbor_traj.max() >= 5000f)
                {
                    var available_index = where1d(current_neighbor_traj.T[0] < 5000f);
                    if ((int)available_index[0] > 0)
                    {
                        current_neighbor_traj[String.Format(":{0}, :", available_index[0])] = current_neighbor_traj[available_index[0]];
                    }
                    if ((int)available_index[-1] < len(current_neighbor_traj))
                    {
                        current_neighbor_traj[String.Format("{0}:, :", available_index[-1])] = current_neighbor_traj[available_index[-1]];
                    }
                }
                neighbor_traj.append(current_neighbor_traj.astype(np.float32));

                if (linear_predict)
                {
                    var pred = predict_linear_for_person(current_neighbor_traj, this.total_frame)[index2];
                    neighbor_traj_linear_pred.append(pred.astype(np.float32));
                }
            }

            this.write_neighbor_traj(neighbor_traj);
            this.write_neighbor_traj_linear_pred(neighbor_traj_linear_pred);
            this.neighbor_number = len(neighbor_agents);
        }

        BaseAgentManager rotate()
        {
            // TODO: rotate in TrainAgentManager
            return this;
        }

        public List<NDArray> get_neighbor_traj()
        {
            var results = new List<NDArray>();
            foreach (var item in this._neighbor_traj)
            {
                results.append(np.array(item).astype(np.float32));
            }
            return results;
        }

        public void clear_all_neighbor_info(){
            this._neighbor_traj = new List<Array>();
            this._neighbor_traj_linear_pred = new List<Array>();
        }

        public void write_neighbor_traj(List<NDArray> neighbor_traj, bool clean = true)
        {
            if (clean)
            {
                this._neighbor_traj = new List<Array>();
            }

            foreach (var item in neighbor_traj)
            {
                this._neighbor_traj.append(item.ToMuliDimArray<float>());
            }
        }

        public void write_neighbor_traj_linear_pred(List<NDArray> neighbor_traj, bool clean = true)
        {
            if (clean)
            {
                this._neighbor_traj_linear_pred = new List<Array>();
            }

            foreach (var item in neighbor_traj)
            {
                this._neighbor_traj_linear_pred.append(item.ToMuliDimArray<float>());
            }
        }

        public List<NDArray> get_neighbor_traj_linear_linear()
        {
            var results = new List<NDArray>();
            foreach (var item in this._neighbor_traj_linear_pred)
            {
                results.append(np.array(item).astype(np.float32));
            }
            return results;
        }

        public (float ade, float fde) calculate_loss(string loss_function = "adefde")
        {
            this.loss = calculate_ADE_FDE_numpy(this.pred, this.groundtruth);
            return this.loss;
        }
    }

    class OnlineAgentManager : BaseAgentManager{
        public int obs_frames;
        public int pred_frames;
        public int wait_frames;
        public int agent_id;
        
        public int linear_predictor; // FIXME linear predictor in onlineAgentManager

        public OnlineAgentManager(
            ArgManagers.OnlineArgsManager args,
            int agent_id,
            int linear_predictor
        ) : base()
        {
            this.obs_frames = args.obs_frames;
            this.pred_frames = args.pred_frames;
            this.wait_frames = args.wait_frames;
            
            this.agent_id = agent_id;
            this.linear_predictor = linear_predictor;
        }

        // TODO getter and setter for traj
        // TODO getter and setter for frame_list
        // TODO getter and setter for pred_linear
    }
}