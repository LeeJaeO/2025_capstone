from vector_database import Vector_database
from retriever import Retriever

vector_database = Vector_database()
embedding_list =vector_database.get_passage_embedding_list()
passage_list = vector_database.get_passage_list()
retriever =Retriever(embedding_list,passage_list)

retriever.dense_dot_similarity("핸드폰을 잃어버렸어요")