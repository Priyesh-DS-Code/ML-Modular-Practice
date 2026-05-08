import os
import logging
from datetime import datetime

file_name=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.logs"
file_path=os.path.join(os.getcwd(), "logs")
os.makedirs(file_path, exist_ok=True)

file_path_name=os.path.join(file_path, file_name)

logging.basicConfig(
    filename=file_path_name,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# if __name__=="__main__":
#     logging.info("Logging has started")



