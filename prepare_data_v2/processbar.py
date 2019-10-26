import sys
import time

class ProcessBar(object):
    """ 
    Attributes:
        length:         {int} total length of process bar
        current_step:   {int} current step
        total_step:     {int} total step
        title           {str} title
    Example:
        import time
        pb = ProcessBar(100)
        for i in range(100):
            pb.step(i)
            time.sleep(0.2)
    """
    def __init__(self, total_step, title='(๑•̀ㅂ•́)و✧ Almost D', length=50):
        
        self.current_step = 0
        self.total_step = total_step
        
        self.start_time = None

        self.length = length
        self.title = title[: 20] + '...' if len(title)>20 else title

    def step(self, current_step=None):
        
        if current_step is not None:
            self.current_step = current_step + 1
        else:
            self.current_step += 1
        
        if self.start_time is None:
            self.start_time = time.time()
        
        fpercent = self.current_step / self.total_step
        ipercent = int(fpercent*self.length//1)
        duration = (time.time() - self.start_time) / 60
        totaltime = duration / fpercent

        bar = "\r{:^s}[{}]ne! [{:3.2%}] >> Elapsed: [{:.2f}]/[{:.2f}] min".\
                    format(self.title, 'o'*ipercent + '.'*(self.length-ipercent), fpercent, 
                        duration, totaltime)
        sys.stdout.write(bar)
        sys.stdout.flush()

        if self.current_step == self.total_step:
            sys.stdout.write('\n')
    
