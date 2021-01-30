/*
 * @Author: Conghao Wong
 * @Date: 2021-01-22 21:00:55
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-29 02:22:01
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
using models.Managers.ArgManagers;
using models.Managers.AgentManagers;

namespace models.Managers
{
    public class MapManager
    {
        public TrainArgsManager args;
        List<TrainAgentManager> agents;

        NDArray void_map;
        NDArray W;
        NDArray b;

        public NDArray guidance_map;

        public MapManager(
            TrainArgsManager args,
            List<TrainAgentManager> agents,
            MapManager init_manager = null
        )
        {
            this.args = args;
            this.agents = agents;

            if (!(init_manager == null))
            {
                this.void_map = init_manager.void_map;
                this.W = init_manager.W;
                this.b = init_manager.b;
            }
            else
            {
                (this.void_map, this.W, this.b) = this.init_guidance_map(agents);
            }
        }

        (NDArray guidance_map, NDArray W, NDArray b) init_guidance_map(List<TrainAgentManager> agents)
        {
            var _traj = this.get_all_trajs_from_agents(agents);
            var traj = np.stack(_traj.ToArray<NDArray>());
            return this._init_guidance_map(traj);
        }

        (NDArray guidance_map, NDArray W, NDArray b) init_guidance_map(List<NDArray> agents)
        {
            var traj = np.stack(agents.ToArray<NDArray>());
            return this._init_guidance_map(traj);
        }

        (NDArray guidance_map, NDArray W, NDArray b) init_guidance_map(NDArray agents)
        {
            return this._init_guidance_map(agents);
        }

        private (NDArray guidance_map, NDArray W, NDArray b) _init_guidance_map(NDArray traj)
        {
            // shape of `traj` should be [*, *, 2] or [*, 2]
            if (len(traj.shape) == 3)
            {
                traj = np.reshape(traj, new int[] { -1, 2 });
            }

            var x_max = np.max(traj[":, 0"]);
            var x_min = np.min(traj[":, 0"]);
            var y_max = np.max(traj[":, 1"]);
            var y_min = np.min(traj[":, 1"]);

            int gm_shape1 = ((x_max - x_min + 2 * this.args.window_size_expand_meter) * this.args.window_size_guidance_map + 1).astype(np.int32);
            int gm_shape2 = ((y_max - y_min + 2 * this.args.window_size_expand_meter) * this.args.window_size_guidance_map + 1).astype(np.int32);
            var guidance_map = np.zeros((gm_shape1, gm_shape2), dtype: np.float32);

            var W = np.array(new float[] { this.args.window_size_guidance_map, this.args.window_size_guidance_map });
            var b = np.array(new float[] { x_min - this.args.window_size_expand_meter, y_min - this.args.window_size_expand_meter });

            return (guidance_map, W, b);
        }

        public NDArray build_guidance_map<T>(List<T> agents, NDArray source = null, bool regulation = true)
        {
            log_function("Building Guidance Map...", end: "\t");

            if (source == null)
            {
                source = this.void_map;
            }

            source = source.copy();
            dynamic _traj;
            if (typeof(T) == typeof(TrainAgentManager))
            {
                _traj = get_all_trajs_from_agents(agents);
            }
            else
            {
                _traj = agents;
            }

            source = this.add_to_map(
                source,
                _traj,
                amplitude: 1,
                radius: 7,
                add_mask: (load_image("./mask_circle.png", return_numpy: true)[":, :, 0"]) / 50f,
                decay: false,
                max_limit: false
            );

            source = np.minimum(source, 30);
            if (regulation)
            {
                source = 1 - source / np.max(source);
            }

            log_function("Done.");
            this.guidance_map = source;
            return source;
        }

        public NDArray build_social_map(TrainAgentManager target_agent, NDArray traj_neighbors = null, NDArray source = null, bool regulation = true)
        {
            if (source == null)
            {
                source = this.void_map;
            }

            source = source.copy();
            var add_mask = load_image("./mask_circle.png", return_numpy: true)[":, :, 0"];

            var trajs = new List<NDArray>();
            var amps = new List<NDArray>();
            var rads = new List<float>();

            // destination
            trajs.append(target_agent.get_pred_linear());
            amps.append(-2 * np.ones((this.args.pred_frames)));
            rads.append(this.args.interest_size);

            // interplay
            var vec_target = target_agent.get_pred_linear()[-1] - target_agent.get_pred_linear()[0];
            for (int i = 0; i < len(traj_neighbors); i++)
            {
                var pred = traj_neighbors[i];
                var vec_neighbor = pred[-1] - pred[0];
                var cosine = linear_activation(
                    calculate_cosine(vec_target, vec_neighbor),
                    a: 1.0f, b: 0.2f
                );
                var velocity = calculate_length(vec_neighbor) / calculate_length(vec_target);

                trajs.append(pred);
                amps.append(-1f * cosine * velocity * np.ones((this.args.pred_frames)).astype(np.float32));
                rads.append(this.args.avoid_size);
            }

            source = this.add_to_map(
                target_map: source,
                trajs_: trajs,
                amplitude: amps,
                radius: rads,
                add_mask: add_mask
            );

            if (regulation)
            {
                if ((float)(np.max(source) - np.min(source)) <= 0.01)
                {
                    source = 0.5 * np.ones_like(source);
                }
                else
                {
                    source = (source - np.min(source)) / (np.max(source) - np.min(source));
                }
            }

            return source;
        }

        public NDArray real2grid(NDArray traj)
        {
            return ((traj - this.b) * this.W).astype(np.int32);
        }

        public NDArray real2grid_paras()
        {
            return np.stack(new NDArray[] { this.W, this.b });
        }

        NDArray add_to_map(
            NDArray target_map, List<NDArray> trajs_,
            dynamic amplitude, dynamic radius, NDArray add_mask = null,
            bool interp = false, bool max_limit = false, bool decay = true
        )
        {
            NDArray trajs = np.stack(trajs_.ToArray());
            if (len(trajs.shape) == 2)
            {
                trajs = np.expand_dims(trajs, axis: 0);
            }

            var n_traj = trajs.shape[0];
            if (!(amplitude.GetType() == typeof(List<NDArray>)))
            {
                amplitude = np.array(amplitude);
            }
            else
            {
                amplitude = np.stack(amplitude.ToArray());
            }

            if (len(amplitude.shape) == 0)
            {
                amplitude = amplitude * np.ones(new int[] { n_traj, trajs.shape[1] }, dtype: np.int32);
            }

            if (!(radius.GetType() == typeof(List<float>)))
            {
                radius = np.array(radius);
            }
            else
            {
                radius = np.array(radius.ToArray());
            }
            if (len(radius.shape) == 0)
            {
                radius = radius * np.ones(new int[] { n_traj }, dtype: np.int32);
            }

            target_map = target_map.copy();
            if (add_mask == null)
            {
                add_mask = np.ones(new int[] { 1, 1 }, dtype: np.int32);
            }


            if (interp)
            {

            }
            else
            {

            }

            foreach (var i in range(n_traj))
            {
                var traj = this.real2grid(trajs[i]);
                var a = amplitude[i];
                var r = (int)(float)(radius[i]);
                var add_mask_ = tf.image.resize(tf.expand_dims(add_mask, axis: -1), (r * 2 + 1, r * 2 + 1)).numpy()[":, :, 0"];
                target_map = this.add_one_traj(target_map, traj, a, r, add_mask_, max_limit: max_limit, amplitude_decay: decay);
            }

            return target_map;
        }

        NDArray add_one_traj(
            NDArray source_map, NDArray traj, NDArray amplitude, NDArray radius, NDArray add_mask,
            bool max_limit = true, bool amplitude_decay = false
        )
        {
            if (amplitude_decay)
            {

            }

            var new_map = np.zeros_like(source_map);
            foreach (var index in range(len(traj)))
            {
                var pos = traj[index];
                var a = len(amplitude) == len(traj) ? amplitude[index] : amplitude;
                if (
                    ((int)(pos[0] - radius) > 0) &&
                    ((int)(pos[1] - radius) > 0) &&
                    ((int)(pos[0] + radius + 1) < new_map.shape[0]) &&
                    ((int)(pos[1] + radius + 1) < new_map.shape[1])
                )
                {
                    new_map[String.Format(
                        "{0}:{1}, {2}:{3}",
                        pos[0] - radius,
                        pos[0] + radius + 1,
                        pos[1] - radius,
                        pos[1] + radius + 1
                    )] = a * add_mask + new_map[String.Format(
                        "{0}:{1}, {2}:{3}",
                        pos[0] - radius,
                        pos[0] + radius + 1,
                        pos[1] - radius,
                        pos[1] + radius + 1
                    )].copy();
                }
            }

            if (max_limit)
            {
                new_map = np.sign(new_map);
            }

            return new_map + source_map;
        }

        public List<NDArray> get_all_trajs_from_agents<T>(List<T> agents)
        {
            var all_trajs = new List<NDArray>();
            foreach (dynamic agent in agents)
            {
                NDArray trajs = agent.get_traj();
                for (int index = 0; index < len(trajs); index++)
                {
                    var traj = trajs[index];
                    all_trajs.append(traj);
                }
            }
            return all_trajs;
        }
    }



}