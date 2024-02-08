import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QGraphicsGridLayout, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QImage, QImageWriter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyqtgraph import PlotWidget, setConfigOptions, mkPen
from PIL import Image, ImageOps
import pyqtgraph.exporters
import imageio

class pulsating_image():
    def __init__(self,image_path,row,col,column_span=1,row_span=1,amplitude=50.,frequency=1,offset = 0.,background = 'white'):
        ##transform image to numpy array
        image = Image.open(image_path)
        # self.height,self.width = image.shape[:2]
        self._row = row
        self._column = col
        self._rowspan = row_span
        self._column_span = column_span
        # Convert the image to a NumPy array
        self.pil_image = image
        self.image_array = np.array(image)
        self.plot = win.addPlot(row=self._row,col=self._column,colspan=self._column_span,rowspan=self._rowspan)
       # self.plot.setFixedSize(self.base_size,self.base_size)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        # self.image.setImage(np.transpose(self.image_array,(1,0,2)))
        self.image.setImage(self.image_array)
        self.background = background
        rotation_angle = 45  # Specify the rotation angle in degrees


        # Hide the axes
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')

        self.amplitude = amplitude  # Pulsating amplitude
        self.frequency = frequency  # Pulsating frequency
        self.offset = offset
        self.phase = 0  # Initial phase
    def update(self):
        # Generate a pulsating intensity
        if self.amplitude != 0.:
            size_factor = self.offset + self.amplitude * np.sin(2 * np.pi * self.frequency * self.phase)
            padded_image = ImageOps.expand(self.pil_image, border=int(200*size_factor), fill=self.background)
        else:
            padded_image = self.pil_image
        # Convert NumPy array to QImage
        self.image.setImage(np.array(padded_image))
        # # Convert NumPy array to QImage
        # qt_image = pg.ImageItem(final_image)
        #
        # # Update the image in the pixmap item
        # self.pixmap_item.setPixmap(pg.QtGui.QPixmap.fromImage(qt_image))

        # Update the phase for the next iteration
        self.phase += 0.05
class image():
    def __init__(self,image_path,row,col,column_span=1,row_span=1,rotation_angle=0):
        ##transform image to numpy array
        image = imageio.imread(image_path)
        self._row = row
        self._column = col
        self._rowspan = row_span
        self._column_span = column_span
        # Convert the image to a NumPy array
        self.image_array = np.array(image)
        self.plot = win.addPlot(row=self._row,col=self._column,colspan=self._column_span,rowspan=self._rowspan)

        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        # self.image.setImage(np.transpose(self.image_array,(1,0,2)))
        self.image.setImage(self.image_array)


        # Hide the axes
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')


class plot():
    def __init__(self,x,y,row,col,title,color=['w'],plottype = 'lineplot',colspan=1, rowspan=1,yticks=None,xrange=None,width=1,xlabel='',ylabel=''):
        self._x = x
        self._current_x = np.zeros(0)

        self._row = row
        self._column = col
        self._rowspan = rowspan
        self._column_span = colspan
        self._title = title
        self._width = width
        self._xlabel = xlabel
        self._ylabel = ylabel
        if type(color) != list:
            self._color = [color]
        else:
            self._color = color
        self.plot = win.addPlot(row=self._row,col=self._column,colspan=self._column_span,rowspan=self._rowspan,title=self._title)
        if yticks is not None:
            self._yticks = yticks
            self.plot.getAxis('left').setTicks(self._yticks)
        if xrange is not None:
            self._xrange = xrange
            self.plot.setXRange(self._xrange[0],self._xrange[1])
        if len(y.shape) == 1:
            self._y = y[:,np.newaxis]
        else:
            self._y = y
        self.lines = []
        self._current_y = []
        for y in range(self._y.shape[1]):
            if plottype == 'lineplot':
                self.lines.append(pg.PlotDataItem(pen=pg.mkPen(self._color[y],width=self._width)))
            elif plottype == 'scatter':
                symbol_path = pg.arrayToQPath(x = np.array([0, 0]), y = np.array([0, 1]))  # Line symbol
                self.lines.append(pg.ScatterPlotItem(size=10))
                self.lines[y].setSymbol(symbol_path)

        # self.line = pg.PlotDataItem(pen=pg.mkPen(self._color))
            self.plot.addItem(self.lines[y])
            self._current_y.append(np.zeros(0))
            self.plot.setLabel('left', self._ylabel)
            self.plot.setLabel('bottom', self._xlabel)
    def update(self,t):
        time = self._x < t
        self._current_x = self._x[time]
        for y in range(self._y.shape[1]):
            self._current_y[y] = self._y[time,y]
            self.lines[y].setData(x=self._current_x,y=self._current_y[y])
        # self._current_y = self._y[time]
        # self.line.setData(x=self._current_x,y=self._current_y)
save_f = "Save/"
freq = 2000
fps = 50
skip_frames = 1
samples = np.load(save_f + "vibr_time.npy")
data = np.load(save_f + "vibr.npy")
spk_rec_hid = np.load(save_f + "spk_rec_hid.npy")
spk_rec = np.load(save_f + "spk_rec.npy")
max_spike = np.load(save_f + "max_spikes.npy")
targets = np.load(save_f + "targets.npy")

app = QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True,size=(1920,1080))
win.setWindowTitle('PyQtGraph Animation Example')
win.setBackground(None)
pg.setConfigOption('background', '#CCDADD')

pg.setConfigOption('foreground', '#3B4A52')
acc_x = plot(x=samples,y=data[:,0,0],row=0,col=0,title='Acc x',color=['#3B4A52'],width=1,xlabel='Time (s)',ylabel='Acceleration (m/s)')
acc_y = plot(x=samples,y=data[:,0,1],row=1,col=0,title='Acc x',color=['#3B4A52'],width=1,xlabel='Time (s)',ylabel='Acceleration (m/s)')
acc_z = plot(x=samples,y=data[:,0,2],row=2,col=0,title='Acc z',color=['#3B4A52'],width=1,xlabel='Time (s)',ylabel='Acceleration (m/s)')

spk_rec_hid_events = np.where(spk_rec_hid[:,0,:])
lif1 = plot(x=spk_rec_hid_events[0]/freq,y=spk_rec_hid_events[1], row=0, col = 1, rowspan=3,title = 'L1',plottype='scatter',yticks=[],xrange=[0,samples.max()],color=['#3B4A52'],xlabel='Time (s)')

chip = image('neuronova_chip.png',row=0,col=2,rotation_angle=90)
#logo = pulsating_image('neuronova_logo.png',row=0,col=2,amplitude=0,frequency=2,offset=0.05,background='white')
spk_rec_events = np.where(spk_rec[:,0,:])
lif2 = plot(x=spk_rec_events[0]/freq,y=spk_rec_events[1], row=1, col = 2, rowspan=1,title = 'L2',plottype='scatter',yticks=[],xrange=[0,samples.max()],color=['#3B4A52'],xlabel='Time (s)')
#load integrate
from scipy import integrate
energy_value = spk_rec_hid[:,0,:].sum(axis=1) + spk_rec[:,0,:].sum(axis=1)
energy_accumulate = np.array([energy_value[:t].sum() for t in range(len(energy_value))])*51e-15 + 10e-15*(spk_rec_hid.shape[2]+spk_rec.shape[2])
energy = plot(x=samples,y=energy_accumulate/1e-12,row=0,col=3,title='Neurons Consumption',plottype='lineplot',rowspan=2,color=['#3B4A52'],width=3,xlabel='Time (s)',ylabel='Energy (pJ)')

bandwidth_value = spk_rec[:,0,:].sum(axis=1)*8*2000
bandwidth = plot(x=samples,y=bandwidth_value/1000,row=2,col=3,title=f'RF Communication',plottype='lineplot',rowspan=1,color=['#3B4A52'],width=3,xlabel='Time (s)',ylabel='Bandwidth (kbps)')

cumulative_spikecount = np.array([spk_rec[:t,0,:].sum(axis=0) for t in range(len(spk_rec))])
winner = np.argmax(cumulative_spikecount,axis=1)
target_pos = np.where(targets[0] == 1)[0]
accuracy_line = np.array([winner[t] == target_pos[0] for t in range(len(winner))]).astype(int)*100
print(accuracy_line)
accuracy = plot(x=samples,y=accuracy_line,row=2,col=2,title='Classification',plottype='lineplot',width=3,color=['#3B4A52'],xlabel='Time (s)',ylabel='Accuracy (%)')
# pg.setConfigOption('background', 'w')

# plot1 = win.addPlot(row=0, col= 0, title='Acceleromter X')
# line1 = pg.PlotDataItem(pen=pg.mkPen('w'))
# plot1.addItem(line1)
#
# plot2 = win.addPlot(row=1, col= 0, title='Acceleromter Y')
# line2 = pg.PlotDataItem(pen=pg.mkPen('w'))
# plot2.addItem(line2)
#
#
# # Initialize data
# x_x = np.zeros(0)
#
# y = np.zeros(0)
# line1.setData(x=x, y=y)
# line2.setData(x=x, y=y)
# data_x = data[:, 0, 0]  # Target values to gradually appear
# data_y = data[:, 0, 1]
# counter = 0
t = 0
frames = []
#import Path
from pathlib import Path
png_for_gif = Path.mkdir(Path.cwd() / 'png_for_gif', exist_ok=True)
def update_plot():
    global t
    acc_x.update(samples[t])
    acc_y.update(samples[t])
    acc_z.update(samples[t])
    lif1.update(samples[t])
    lif2.update(samples[t])
    energy.update(samples[t])
    bandwidth.update(samples[t])
    accuracy.update(samples[t])
    #chip.update()
    t += skip_frames
    print(t)
    # global y_x,x_x,y_y,x_y
    # if len(y_x) < len(data_x):
    #     x_x = np.concatenate([x_x, samples[len(y_x):len(y_x)+10]])
    #     y_x = np.concatenate([y_x, data_x[len(y_x):len(y_x) + 10]])
    #     # y = np.clip(y + np.random.normal(0, 0.1, size=len(y)), 0, 1)  # Update the data (example: add random noise)
    # line1.setData(x=x_x,y=y_x)  # Update the scatter plot data
    # if len(y_y) < len(data_x):
    #     x_y = np.concatenate([x_y, samples[len(y_y):len(y_y)+10]])
    #     y_y = np.concatenate([y_y, data_x[len(y_y):len(y_y) + 10]])
    #     # y = np.clip(y + np.random.normal(0, 0.1, size=len(y)), 0, 1)  # Update the data (example: add random noise)
    # line2.setData(x=x_y,y=y_y)  # Update the scatter plot data
    # Capture the current state of the window as an image
    # Capture the current state of the window as an image
    exporter = pg.exporters.ImageExporter(win.scene())

    width = 1920
    exporter.parameters()['width'] = width
    exporter.parameters()['height'] = width/2.5
    exporter.parameters()['antialias'] = False
    exporter.parameters()['background'] = '#CCDADD'
    pg.setConfigOption('foreground', 'k')

    # Save to animated GIF
    exporter.export(f'png_for_gif/{t}.png')



    # img = win.grab()
    # image = img.toImage()
    # frame_data = image.bits().asstring(image.byteCount())  # Convert QImage to bytes
    # frames.append(frame_data)

    if t >= len(samples):
    # if t>200:
        # Save the frames as an animated GIF using Matplotlib
        #
        timer.stop()
        print("Done")
        listfiles = sorted(Path('png_for_gif').glob('*.png'),key = lambda x: int(x.stem))
        print(listfiles)
        import imageio
        import imageio.v2 as iio
        w = iio.get_writer('my_video.mp4', format='FFMPEG', mode='I', fps=fps)
        for filename in listfiles:
            w.append_data(imageio.imread(filename))
        w2 = iio.get_writer('my_video.gif', format='GIF', mode='I')
        for filename in listfiles:
            w2.append_data(imageio.imread(filename))
        w.close()
        # writer = imageio.get_writer('test.fig', fps=fps)
        # for filename in listfiles:
        #     writer.append_data(imageio.imread(filename))
        # writer.close()
        # sys.exit(app.exec_())


timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(100)  # Set the timer interval in milliseconds

if __name__ == '__main__':
    sys.exit(app.exec_())
