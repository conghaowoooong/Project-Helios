/*
 * @Author: Conghao Wong
 * @Date: 2021-01-27 17:18:51
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-05 23:53:25
 * @Description: file content
 */

using System;
using System.IO;
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
using System.Collections;
using modules.models.Base;


namespace modules.models.Prediction
{
    public class EntireTrajectory
    {
        int _agent_index;
        NDArray _traj;
        List<Array> _video_neighbor_list;
        NDArray _frame_list;
        int _start_frame;
        public int _end_frame;

        public int agent_index
        {
            get
            {
                return this._agent_index;
            }
        }
        public NDArray traj
        {
            get
            {
                return this._traj;
            }
        }
        public List<Array> video_neighbor_list
        {
            get
            {
                return this._video_neighbor_list;
            }
        }
        public NDArray frame_list
        {
            get
            {
                return this._frame_list;
            }
        }
        public int start_frame
        {
            get
            {
                return this._start_frame;
            }
        }
        public int end_frame
        {
            get
            {
                return this._end_frame;
            }
        }

        public EntireTrajectory(int agent_index, List<Array> video_neighbor_list, NDArray video_matrix, NDArray frame_list, float init_position)
        {
            this._agent_index = agent_index;
            this._traj = np.transpose(video_matrix, new int[] { 1, 0, 2 })[agent_index].copy();
            this._video_neighbor_list = video_neighbor_list;
            this._frame_list = frame_list;

            var base_ = this.traj.T[0];
            var diff = base_[":-1"] - base_["1:"];

            var appear = where1d(diff > init_position / 2.0f);
            var disappear = where1d(diff < -init_position / 2.0f);

            this._start_frame = len(appear) > 0 ? appear[0] + 1 : 0;
            this._end_frame = len(disappear) > 0 ? disappear[0] + 1 : len(base_);
        }
    }


    public class DatasetManager
    {
        TrainArgs args;
        public string dataset_name;
        Dataset dataset_info;
        List<Array> video_neighbor_list;
        NDArray video_matrix;
        NDArray frame_list;
        public int person_number;
        List<EntireTrajectory> all_entire_trajectories;

        public DatasetManager(TrainArgs args, string dataset_name, Tuple<List<Array>, NDArray, NDArray> custom_list = null)
        {
            this.args = args;
            this.dataset_name = dataset_name;
            this.dataset_info = new PredictionDatasetManager().call(dataset_name);

            if (custom_list != null && len(custom_list) == 3)
            {
                this.video_neighbor_list = custom_list.Item1;
                this.video_matrix = custom_list.Item2;
                this.frame_list = custom_list.Item3;
            }
            else
            {
                (this.video_neighbor_list, this.video_matrix, this.frame_list) = this.load_data();
            }

            this.all_entire_trajectories = this.prepare_agent_data();
        }

        (List<Array> video_neighbor_list, NDArray video_matrix, NDArray frame_list) load_data()
        {
            List<Array> video_neighbor_list;
            NDArray video_matrix;
            NDArray frame_list;

            dir_check("./dataset_npz");
            var base_path = dir_check(String.Format("./dataset_npz/{0}", this.dataset_name));
            var npy_path = String.Format("{0}/data.npz", base_path);
            var dat_path = new string[] {
                String.Format("{0}/neighbor_list.dat", base_path),
                String.Format("{0}/video.npy", base_path),
                String.Format("{0}/frame.npy", base_path),
            };

            if (file_exist(dat_path[0]) && file_exist(dat_path[1]) && file_exist(dat_path[2]))
            {
                video_neighbor_list = read_file<List<Array>>(dat_path[0]);
                video_matrix = np.load(dat_path[1]);
                frame_list = np.load(dat_path[2]);

                return (video_neighbor_list, video_matrix, frame_list);

            }
            else
            {
                (var person_data, var frame_list_) = this.load_from_csv_dataset(this.dataset_name);
                frame_list = np.array(frame_list_);
                var person_list = (List<int>)person_data.Keys.ToList();

                var person_number = len(person_list);
                var frame_number = len(frame_list);

                video_matrix = this.args.init_position * np.ones(new int[] { frame_number, person_number, 2 }, dtype: np.float32);
                var person_dict = make_dict(person_data.Keys.ToArray(), np.arange(len(person_data.Keys.ToList())).ToArray<int>());
                var frame_dict = make_dict(frame_list.ToArray<int>(), np.arange(len(frame_list)).ToArray<int>());

                log_function(String.Format("Solving dataset {0}...", this.dataset_name), end: "\t");
                foreach (var person in person_data.Keys)
                {
                    var person_index = person_dict[person];
                    var frame_list_current = person_data[person].T[0].astype(np.int32);

                    var frame_index_current_list = new List<int>();
                    foreach (int frame_current in frame_list_current)
                    {
                        frame_index_current_list.append(frame_dict[frame_current]);
                    }
                    var frame_index_current = np.array(frame_index_current_list);
                    video_matrix[frame_index_current, person_index] = person_data[person][":, 1:"].copy();
                }
                log_function("Done.");

                var video_neighbor_list_ = new List<Array>();
                foreach (var index in range(len(video_matrix)))
                {
                    var data = video_matrix[index];
                    video_neighbor_list_.append(where1d(!(data.T[0] == this.args.init_position)).ToArray<int>());
                }

                write_file(String.Format("{0}/neighbor_list.dat", base_path), video_neighbor_list_);
                np.save(dat_path[1], video_matrix);
                np.save(dat_path[2], frame_list);

                return (video_neighbor_list_, video_matrix, frame_list);
            }
        }

        (Dictionary<int, NDArray> person_data, List<int> frame_list) load_from_csv_dataset(string dataset_name)
        {
            var dataset_dir_current = this.dataset_info.dataset_dir;
            var order = this.dataset_info.order;

            var csv_file_path = String.Format("{0}/true_pos_.csv", dataset_dir_current);
            var data = csv2ndarray(csv_file_path).T;

            // load data, and sort by person id
            var person_data = new Dictionary<int, NDArray>();
            var person_list = data.T["1, :"].astype(np.int32).ToArray<int>().ToList().Distinct().ToList();

            foreach (var person in person_list)
            {
                var index_current = where1d(data.T["1, :"] == person);
                var temp = data[index_current];
                var slice_index = np.array(new int[] { 0, 2 + order[0], 2 + order[1] });

                person_data[person] = temp.T[slice_index].T.astype(np.float32);
            }

            var frame_list = data.T["0, :"].astype(np.int32).ToArray<int>().ToList().Distinct().ToList();

            return (person_data, frame_list);
        }

        List<EntireTrajectory> prepare_agent_data()
        {
            this.person_number = this.video_matrix.shape[1];
            var all_entrie_trajectories = new List<EntireTrajectory>();
            foreach (var person in range(this.person_number))
            {
                all_entrie_trajectories.append(
                    new EntireTrajectory(
                        person,
                       this.video_neighbor_list,
                       this.video_matrix,
                       this.frame_list,
                       this.args.init_position
                    )
                );
            }
            return all_entrie_trajectories;
        }

        TrainAgentManager _get_trajectory(
            int agent_index, int start_frame, int obs_frame, int end_frame,
            int frame_step = 1,
            bool add_noise = false
        )
        {
            var trajecotry_current = this.all_entire_trajectories[agent_index];
            var frame_list = trajecotry_current.frame_list;
            var neighbor_list = new List<int>();
            foreach (int neighbor in trajecotry_current.video_neighbor_list[obs_frame - frame_step])
            {
                if (neighbor != agent_index)
                {
                    neighbor_list.append(neighbor);
                }
            }

            var neighbor_agents = new List<EntireTrajectory>();
            foreach (var item in neighbor_list)
            {
                neighbor_agents.append(this.all_entire_trajectories[item]);
            }

            return new TrainAgentManager().init_data(
                trajecotry_current, neighbor_agents,
                frame_list, start_frame, obs_frame, end_frame,
                frame_step: frame_step, add_noise: add_noise
            );
        }

        public List<TrainAgentManager> sample_train_data()
        {
            var sample_rate = this.dataset_info.paras[0];
            var frame_rate = this.dataset_info.paras[1];
            var frame_step = (int)(0.4f / (sample_rate / frame_rate));
            var timebar = new LogFunction.LogBar();

            // sample all train agents
            var train_agents = new List<TrainAgentManager>();
            log_function("Prepare train data...", end: "\t");
            timebar.clean();
            foreach (var agent_id in range(this.person_number))
            {
                var trajecotry_current = this.all_entire_trajectories[agent_id];
                var start_frame = trajecotry_current.start_frame;
                var end_frame = trajecotry_current.end_frame;

                for (int frame_point = start_frame; frame_point < end_frame; frame_point += (frame_step) * this.args.step)
                {
                    if (frame_point + (this.args.obs_frames + this.args.pred_frames) * frame_step > end_frame)
                    {
                        break;
                    }

                    train_agents.append(this._get_trajectory(
                        agent_id, frame_point,
                        frame_point + this.args.obs_frames * frame_step,
                        frame_point + (this.args.obs_frames + this.args.pred_frames) * frame_step,
                        frame_step: frame_step, add_noise: false
                    ));
                }
                timebar.log(agent_id, this.person_number);
            }

            // write maps
            var map_manager = new MapManager(this.args, train_agents);
            map_manager.build_guidance_map(agents: train_agents);
            np.save(String.Format("./dataset_npz/{0}/gm.npy", this.dataset_name), map_manager.guidance_map);

            log_function("Building Social Map...", end: "\t");
            timebar.clean();
            foreach (var index in range(len(train_agents)))
            {
                train_agents[index].trajMapManager = map_manager;
                if (!(train_agents[index].neighbor_number == 0))
                {
                    train_agents[index].socialMapManager = map_manager.build_social_map(
                        target_agent: train_agents[index],
                        traj_neighbors: np.stack(train_agents[index].get_neighbor_traj_linear_linear().ToArray())
                    );
                }
                timebar.log(index, len(train_agents));
            }

            return train_agents;
        }
    }

    public class TrainDataManager
    {
        TrainArgs args;
        PredictionDatasetManager dataset_info;
        List<string> dataset_list;
        List<string> train_list;
        List<string> val_list;
        public Dictionary<string, object> train_info;

        public TrainDataManager(
            TrainArgs args, bool save = true,
            string prepare_type = "all"
        )
        {
            this.args = args;
            this.dataset_info = new PredictionDatasetManager();

            this.dataset_list = this.dataset_info.dataset_list[this.args.dataset];
            var test_list = this.dataset_info.dataset_list[String.Format("{0}test", this.args.dataset)];

            if (this.args.dataset == "ethucy")
            {
                this.train_list = new List<string>();
                foreach (var item in this.dataset_list)
                {
                    this.train_list.append(item);
                }
                this.val_list = new List<string> { this.args.test_set };
            }
            else if (this.args.dataset == "sdd")
            {

            }
            else
            {

            }

            if (prepare_type == "all")
            {
                this.train_info = this.get_train_and_test_agents();
            }
            else if (prepare_type == "test")
            {
                int count = 1;
                foreach (var dataset in test_list)
                {
                    log_function(String.Format("Preparing {0}/{1}...", count, len(test_list)));
                    this.prepare_train_files(new List<DatasetManager> { new DatasetManager(this.args, dataset) });
                }
            }
            else
            {

            }
        }

        Dictionary<string, object> get_train_and_test_agents()
        {
            dir_check("./dataset_npz/");
            var datasets = this.dataset_info.dataset_list[this.args.dataset];

            // prepare train agents
            var data_managers_train = new List<DatasetManager>();
            var sample_number_original = 0;
            var sample_time = 0;
            foreach (var dataset in this.train_list)
            {
                var dm = new DatasetManager(this.args, dataset);
                data_managers_train.append(dm);
                sample_number_original += dm.person_number;
            }

            // prepare test agetns
            var data_managers_test = new List<DatasetManager>();
            foreach (var dataset in this.val_list)
            {
                var dm = new DatasetManager(this.args, dataset);
                data_managers_test.append(dm);
            }

            // prepare test and train data
            var test_agents = this.prepare_train_files(data_managers_test);
            var train_agents = this.prepare_train_files(data_managers_train);
            sample_time = len(train_agents);

            return new Dictionary<string, object> {
                {"train_data", train_agents},
                {"test_data", test_agents},
                {"train_number", len(train_agents)},
                {"sample_time", sample_time}
            };
        }

        public void zip_and_save(string save_dir, List<TrainAgentManager> agents){
            var save_dict = new Dictionary<string, object>();
            foreach (int index in range(len(agents))){
                save_dict[String.Format("{0}", index)] = agents[index].zip_data();
            }
            write_file(save_dir, save_dict);
        }

        public List<TrainAgentManager> load_and_unzip(string save_dir){
            var save_dict = read_file<Dictionary<string, object>>(save_dir);
            List<TrainAgentManager> results = new List<TrainAgentManager>();
            foreach (var val in save_dict.Values){
                var am = new TrainAgentManager();
                results.append((TrainAgentManager)am.load_data((Dictionary<string, object>)val));
            }
            return results;
        }

        public List<TrainAgentManager> prepare_train_files(
            List<DatasetManager> dataset_managers, string mode = "test"
        )
        {
            var all_agents = new List<TrainAgentManager>();
            var count = 1;
            List<TrainAgentManager> agents;

            foreach (var dm in dataset_managers)
            {
                log_function(String.Format("({0}/{1})  Prepare test data in `{2}`...", count, len(dataset_managers), dm.dataset_name));
                var path = String.Format("./dataset_npz/{0}/agent.dat", dm.dataset_name);
                if (!file_exist(path))
                {
                    agents = dm.sample_train_data();
                    this.zip_and_save(path, agents);
                }
                else
                {
                    agents = this.load_and_unzip(path);
                }

                if (mode == "train")
                {

                }

                all_agents = all_agents.concat(agents).ToList();
                count += 1;
            }

            return all_agents;
        }
    }
}