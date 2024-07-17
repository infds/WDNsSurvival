
import os
from datetime import datetime
import pickle


class BaseModel():


    def __init__(self,
                 model_save_path: str = "Results",
                 ) -> None:

        self.model_save_path = model_save_path
        self.global_SHAP_save_path = os.path.join(self.model_save_path, "Explainers")
        self.predict_save_path = os.path.join(self.model_save_path, "Models")

    def get_model_subdir(self,model_type:str)->str:
        valid_paths = ["Predict", "Global_Explainers", "Instances_Explainers"]
        if model_type not in valid_paths:
            raise ValueError(f"Invalid model_save_type")

        if model_type == "Predict":
            return self.predict_save_path
        elif model_type=="Global_Explainers":
            return self.global_SHAP_save_path

    def model_save(self, TrainedModel, Model_type) -> None:
        """
        提供模型保存功能
        :param TrainedModel: 保存训练好的模型
        :param Model_type:
        :return:
        """
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        current_time = datetime.now().strftime("%Y%m%d-%H")
        model_subdir=self.get_model_subdir(Model_type)
        if Model_type == "Predict":
            os.mkdir(model_subdir)
            file_name = f"Trained_Model_{current_time}.pkl"
            with open(os.path.join(model_subdir, file_name), "wb") as f:
                pickle.dump(TrainedModel, f)

        elif Model_type == "Global_Explainers":
            os.mkdir(model_subdir)
            file_name = f"Global_SurvSHAP_Model_{current_time}.pkl"
            with open(os.path.join(model_subdir, file_name), "wb") as f:
                pickle.dump(TrainedModel, f)
        else:
            os.mkdir(model_subdir)
            file_name = f"Instances_SurvSHAP_Model_{current_time}.pkl"
            with open(os.path.join(model_subdir, file_name), "wb") as f:
                pickle.dump(TrainedModel, f)
        print("模型已保存")


    def get_latest_model(self, model_type: str):
        model_subdir = self.get_model_subdir(model_type)
        if not os.path.exists(model_subdir):
            print("模型文件夹不存在，需要训练新模型...")

        model_files = [f for f in os.listdir(model_subdir) if f.endswith('.pkl')]
        if not model_files:
            print("没有找到已保存的模型，需要训练新模型...")

        latest_model_file = max(model_files,
                                key=lambda x: datetime.strptime(x.split('_')[-1].split('.')[0], "%Y%m%d-%H"))
        latest_model_path = os.path.join(model_subdir, latest_model_file)

        with open(latest_model_path, "rb") as f:
            model = pickle.load(f)
        print(f"已加载模型: {latest_model_path}")
        return model