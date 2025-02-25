from openai import OpenAI
import numpy as np

class Retriever:
    def __init__(self,passage_embedding_list,passage_list):
        self.passage_embedding_list = passage_embedding_list
        self.passage_list = passage_list


    def dense_dot_similarity(self, query):
        client = OpenAI(
            api_key="up_2W4biq7NxCEuhRu9e6HhXEw6mLUXi",
            base_url="https://api.upstage.ai/v1/solar"
        )
        query_embedding = client.embeddings.create(
            model="embedding-query",
            input=query
        ).data[0].embedding

        # 2️⃣ 모든 패시지와 쿼리의 내적 값 계산
        similarity_list = [np.dot(passage_embedding, query_embedding) for passage_embedding in self.passage_embedding_list]

        # 3️⃣ 유사도 내림차순으로 상위 5개 인덱스 가져오기
        top_k = 5
        top_indices = np.argsort(similarity_list)[-top_k:][::-1]  # 가장 큰 값부터 정렬

        # 4️⃣ 상위 5개 유사 결과 추출
        top_passages = [self.passage_list[idx] for idx in top_indices]
        top_similarities = [similarity_list[idx] for idx in top_indices]

        # 5️⃣ 결과 반환 (문서와 유사도 함께)
        return list(zip(top_passages, top_similarities))
    
    #def sparse_bm25(self,query):

    #def esemble_dense_sparse(self):
