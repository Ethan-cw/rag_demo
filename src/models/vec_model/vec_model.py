import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from transformers import BertTokenizer

from src.models.vec_model.m3e_model import M3EModel


class VectorizeModel:
    def __init__(self, ptm_model_path, device="cpu") -> None:
        self.tokenizer = BertTokenizer.from_pretrained(ptm_model_path)
        self.model = M3EModel(pretrained_bert_path=ptm_model_path, pooling="cls")
        self.model.eval()

        # self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.DEVICE = device
        logger.info(device)
        self.model.to(self.DEVICE)

        self.pdist = nn.PairwiseDistance(2)

    def predict_vec(self, query):
        q_id = self.tokenizer(query, max_length=200, truncation=True, padding="max_length", return_tensors='pt')
        with torch.no_grad():
            q_id_input_ids = q_id["input_ids"].squeeze(1).to(self.DEVICE)
            q_id_attention_mask = q_id["attention_mask"].squeeze(1).to(self.DEVICE)
            q_id_token_type_ids = q_id["token_type_ids"].squeeze(1).to(self.DEVICE)
            q_id_pred = self.model(q_id_input_ids, q_id_attention_mask, q_id_token_type_ids)

        return q_id_pred

    def predict_vec_request(self, query):
        q_id_pred = self.predict_vec(query)
        return q_id_pred.cpu().numpy().tolist()

    def predict_sim(self, q1, q2):
        q1_v = self.predict_vec(q1)
        q2_v = self.predict_vec(q2)
        sim = F.cosine_similarity(q1_v[0], q2_v[0], dim=-1)
        return sim.numpy().tolist()


if __name__ == "__main__":
    VEC_MODEL_PATH = "D:\\pyproject\\LLM\\m3e-base"
    vec_model = VectorizeModel(VEC_MODEL_PATH)
    print(vec_model.predict_vec("什么人不能吃花生"))
