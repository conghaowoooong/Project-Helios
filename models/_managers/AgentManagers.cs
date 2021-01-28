/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 20:04:30
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-28 19:25:20
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

namespace models.Managers.AgentManagers{
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

        public Array frame_list;
        public Array frame_list_future;

        public void copy(){
            
        }

        // 001
        public NDArray get_traj(){
            return np.array(this._traj).astype(np.float32);
        }

        // 002
        public NDArray get_pred(){
            return np.array(this._traj_pred).astype(np.float32);
        }

        // 003
        public NDArray get_pred_linear(){
            return np.array(this._traj_pred_linear).astype(np.float32);
        }

        // 004
        public NDArray get_future_traj(){
            return np.array(this._traj_future).astype(np.float32);
        }

        // 005
        public NDArray get_traj_map(){
            return np.array(this._traj_map).astype(np.float32);
        }

        // 006
        public NDArray get_social_map(){
            return np.array(this._social_map).astype(np.float32);
        }

        // 007
        public NDArray get_map(){
            if (this._social_map == null){
                return 0.5 * this.get_traj_map();
            } else {
                return 0.5 * this.get_social_map() + 0.5 * this.get_traj_map();
            }
        }

        // 008
        public NDArray get_frame_list(){
            return np.concatenate(new NDArray[] {
                np.array(this.frame_list), 
                np.array(this.frame_list_future)
            });
        }

        // 201
        public void write_traj(NDArray _traj, NDArray frame_list){
            this._traj = _traj.astype(np.float32).ToMuliDimArray<float>();
            this.frame_list = frame_list.ToArray<int>();
        }

        // 202
        public void write_pred(NDArray pred){
            this._traj_pred = pred.astype(np.float32).ToMuliDimArray<float>();
        }

        // 203
        public void write_pred_linear(NDArray pred){
            this._traj_pred_linear = pred.astype(np.float32).ToMuliDimArray<float>();
        }

        // 204
        public void write_future_traj(NDArray _traj, NDArray frame_list = null){
            this._traj_future = _traj.astype(np.float32).ToMuliDimArray<float>();
            this.frame_list_future = frame_list.ToArray<int>();
        }

        // 205
        public void write_traj_map(MapManager trajmap){
            var full_map = trajmap.guidance_map;
            var half_size = trajmap.args.map_half_size;
            var center_pos = trajmap.real2grid(this.get_traj()["-1, :"]);
            
            var original_map = full_map[String.Format("{0}, {1}", 
                models.HelpMethods.get_slice_index(np.maximum(center_pos[0]-half_size, 0), np.minimum(center_pos[0]+half_size, full_map.shape[0])),
                models.HelpMethods.get_slice_index(np.maximum(center_pos[1]-half_size, 0), np.minimum(center_pos[1]+half_size, full_map.shape[1]))
            )];
            var final_map = tf.image.resize(tf.expand_dims(original_map, axis:-1), (2*half_size, 2*half_size));
            
            this._traj_map = final_map.numpy()[":, :, 0"].astype(np.float32).ToMuliDimArray<float>();
            this._real2grid = trajmap.real2grid_paras().astype(np.float32).ToMuliDimArray<float>();
        }

        // 206
        public void write_social_map(MapManager trajmap, NDArray full_map){
            var half_size = trajmap.args.map_half_size;
            var center_pos = trajmap.real2grid(this.get_traj()["-1, :"]);
            
            var original_map = full_map[String.Format("{0}, {1}", 
                models.HelpMethods.get_slice_index(np.maximum(center_pos[0]-half_size, 0), np.minimum(center_pos[0]+half_size, full_map.shape[0])),
                models.HelpMethods.get_slice_index(np.maximum(center_pos[1]-half_size, 0), np.minimum(center_pos[1]+half_size, full_map.shape[1]))
            )];
            var final_map = tf.image.resize(tf.expand_dims(original_map, axis:-1), (2*half_size, 2*half_size));
            
            this._social_map = final_map.numpy()[":, :, 0"].astype(np.float32).ToMuliDimArray<float>();
            this._real2grid = trajmap.real2grid_paras().astype(np.float32).ToMuliDimArray<float>();
        }
    }


    [Serializable]
    public class TrainAgentManager : BaseAgentManager{
        List<Array> _neighbor_traj;
        List<Array> _neighbor_traj_linear_pred;
        bool linear_predict;
        public int obs_length;
        public int total_frame;
        public int neighbor_number;
        (double ade, double fde) loss;

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
        ){
            this.linear_predict = linear_predict;

            // trajectory info
            this.obs_length = (obs_frame - start_frame) / frame_step;
            this.total_frame = (end_frame - start_frame) / frame_step;

            // Trajectory
            var whole_traj = target_agent.traj[String.Format("{0}:{1}:{2}", start_frame, end_frame, frame_step)].copy();
            var frame_list_current = np.array(frame_list)[String.Format("{0}:{1}:{2}", start_frame, end_frame, frame_step)].copy();
            
            if (add_noise){

            }
            
            var index1 = String.Format(":{0}", this.obs_length);
            var index2 = String.Format("{0}:", this.obs_length);
            this.write_traj(whole_traj[index1], frame_list_current[index1]);
            this.write_future_traj(whole_traj[index2], frame_list_current[index2]);

            if (linear_predict){
                this.write_pred_linear(predict_linear_for_person(
                    this.get_traj(),
                    time_pred:this.total_frame
                )[index2]);
            }

            // neighbor info
            var neighbor_traj = new List<NDArray>();
            var neighbor_traj_linear_pred = new List<NDArray>();
            foreach (var neighbor in neighbor_agents){
                var current_neighbor_traj = neighbor.traj[String.Format("{0}:{1}:{2}", start_frame, end_frame, frame_step)].copy();
                if (current_neighbor_traj.max() >= 5000f){
                    var available_index = where1d(current_neighbor_traj.T[0] < 5000f);
                    if ((int)available_index[0] > 0){
                        current_neighbor_traj[String.Format(":{0}, :", available_index[0])] = current_neighbor_traj[available_index[0]];
                    }
                    if ((int)available_index[-1] < len(current_neighbor_traj)){
                        current_neighbor_traj[String.Format("{0}:, :", available_index[-1])] = current_neighbor_traj[available_index[-1]];
                    }
                }
                neighbor_traj.append(current_neighbor_traj.astype(np.float32));

                if (linear_predict){
                    var pred = predict_linear_for_person(current_neighbor_traj, this.total_frame)[index2];
                    neighbor_traj_linear_pred.append(pred.astype(np.float32));
                }
            }

            this.write_neighbor_traj(neighbor_traj);
            this.write_neighbor_traj_linear_pred(neighbor_traj_linear_pred);
            this.neighbor_number = len(neighbor_agents);
        }

        BaseAgentManager rotate(){
            return this;
        }

        public void write_neighbor_traj(List<NDArray> neighbor_traj, bool clean = true){
            if (clean){
                this._neighbor_traj = new List<Array>();
            }
            
            foreach (var item in neighbor_traj){
                this._neighbor_traj.append(item.ToMuliDimArray<float>());
            }
        }

        public void write_neighbor_traj_linear_pred(List<NDArray> neighbor_traj, bool clean = true){
            if (clean){
                this._neighbor_traj_linear_pred = new List<Array>();
            }
            
            foreach (var item in neighbor_traj){
                this._neighbor_traj_linear_pred.append(item.ToMuliDimArray<float>());
            }
        }

        public List<NDArray> get_neighbor_traj(){
            var results = new List<NDArray>();
            foreach (var item in this._neighbor_traj){
                results.append(np.array(item).astype(np.float32));
            }
            return results;
        }

        public List<NDArray> get_neighbor_traj_linear_pred(){
            var results = new List<NDArray>();
            foreach (var item in this._neighbor_traj_linear_pred){
                results.append(np.array(item).astype(np.float32));
            }
            return results;
        }

        public (double ade, double fde) calculate_loss(string loss_function="adefde"){
            this.loss = calculate_ADE_FDE_numpy(this.get_pred(), this.get_future_traj());
            return this.loss;
        }
    }
}