from openai import OpenAI


# Person 클래스 정의
class Vector_database:
    def __init__(self):
        with open("concierge.txt", "r", encoding="utf-8") as file:
            content = file.read()

        concierge = content.strip().split("\n\n")

        print("concierge 문서 개수: "+str(len(concierge)))
        #######################################################
        with open("F&B.txt", "r", encoding="utf-8") as file:
            content = file.read()

        f_b = content.strip().split("\n\n")

        print("F&B 문서 개수: "+str(len(f_b)))
        #######################################################

        with open("facility.txt", "r", encoding="utf-8") as file:
            content = file.read()

        facility = content.strip().split("\n\n")

        print("facility 문서 개수: "+str(len(facility)))
        #######################################################
        with open("frontDesk.txt", "r", encoding="utf-8") as file:
            content = file.read()

        frontdesk = content.strip().split("\n\n")

        print("frontDesk 문서 개수: "+str(len(frontdesk)))
        #######################################################
        with open("housekeeping.txt", "r", encoding="utf-8") as file:
            content = file.read()

        housekeeping = content.strip().split("\n\n")

        print("housekeeping 문서 개수: "+str(len(housekeeping)))

        documents = concierge+f_b+facility+frontdesk+housekeeping
        #################################################################

        client = OpenAI(
            api_key="up_2W4biq7NxCEuhRu9e6HhXEw6mLUXi",
            base_url="https://api.upstage.ai/v1/solar"
        )

        passage_list = documents
        self.passage_list = passage_list

        batch_process_resut_1 = client.embeddings.create(
            model = "embedding-passage",
            input = passage_list[0:100]
        ).data
        
        passage_embedding_list_1 = [i.embedding for i in batch_process_resut_1]

        batch_process_resut_2 = client.embeddings.create(
            model = "embedding-passage",
            input = passage_list[100:]
        ).data
        
        passage_embedding_list_2 = [i.embedding for i in batch_process_resut_2]
        passage_embedding_list = passage_embedding_list_1+passage_embedding_list_2
        self.passage_embedding_list =passage_embedding_list

    def get_passage_embedding_list(self):
        return self.passage_embedding_list
    
    def get_passage_list(self):
        return self.passage_list
    

