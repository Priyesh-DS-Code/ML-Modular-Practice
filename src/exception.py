import sys
from src.logger import logging

def error_message_details(error_msg, error_details:sys):
    _,_,exc_tb=error_details.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename
    error_msg="The error occured in file [{0}] line number [{1}] and the message is [{2}]".format(
        filename,
        exc_tb.tb_lineno,
        str(error_msg)
    )

    return error_msg


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error=error_message_details(error_msg=error_message, error_details=error_detail)

    def __str__(self):
        return self.error

# try:
#     a=1/0
# except Exception as e:
#     logging.info("Zero Devision Error")
#     raise CustomException(e, sys)
    


