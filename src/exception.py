import sys

def error_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message='there is an error in python script [{1}] line number [{2}] error name [{3}]'.format(file_name,exc_tb.tb_lineno,str(error))
    return error_message
    
class custom_exception(Exception):
    def __init__(self,error_message,error_detail:sys):
       super().__init__(error_message)
       self.error=error_detail(error_message,error_detail=error_detail)
       
    def __str__(self):
        return self.error_message