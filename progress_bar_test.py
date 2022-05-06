import progressbar
from time import sleep


asd = 5
bar = progressbar.ProgressBar(
    maxval=20,
    widgets=['Train: ', progressbar.Counter(), '/20 ', progressbar.Bar('=','[',']'), progressbar.Percentage()],
)
bar2 = progressbar.ProgressBar(
    maxval=10,
    widgets=['Val: ', progressbar.Counter(), '/10 ', progressbar.Bar('=','[',']'), progressbar.Percentage()],
)

# for u in range(5):
#     bar.start()
#     for i in range(20):
#         if(i%5==4):
#             bar2.start()
#             for j in range(10):
#                 bar2.update(j+1)
#                 sleep(0.3)
#             bar2.finish()
#         bar.update(i+1)
#         sleep(0.2)
#     bar.finish()
# bar.start()
# for i in range(20):
#         bar.update(i+1)
#         sleep(0.2)
bar = progressbar.ProgressBar(
    widgets=[]
)
bar.start()
for i in range(10):
        bar.update(i+1)
        sleep(0.1)
bar.finish()
bar = progressbar.ProgressBar(
    maxval=20,
    widgets=['Train: ', progressbar.Counter(), '/20 ', progressbar.Bar('=','[',']'), progressbar.Percentage(), '    ', progressbar.Timer(), '    Loss: ', ''],
)
bar.start()
for i in range(10):
        bar.widgets[-1] = '50'
        bar.update(i+1)
        sleep(0.1)
bar.finish()