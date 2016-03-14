import pyglet
from pyglet.window import key
from lib.Board import Board
from lib.Game import RLGame
from lib.Learning import QLearner, SarsaLambdaLearner
from lib.util import Scoreplot
import cPickle as pickle
import glob

global lastgame
global lastscores
lastgame = 0
lastscores = [0]

DRAW = False
UPDATE_INTERVAL = 0.1
PLOT = False
iteration = 0
maxscore = 0

BLOCK_IMG_FILE = 'img/block.png'

ACTIONNAMES = {
    -1: 'Do nothing',
    key.MOTION_LEFT: 'Move Left',
    key.MOTION_RIGHT: 'Move Right',
    key.MOTION_UP: 'Rotate',
    key.MOTION_DOWN: 'Move Down',
}

# these are the dimensions from the gameboy version
BOARD_WIDTH = 6
BOARD_HEIGHT = 12

block = pyglet.image.load(BLOCK_IMG_FILE)
block_sprite = pyglet.sprite.Sprite(block)

BLOCK_WIDTH = block.width
BLOCK_HEIGHT = block.height

if DRAW:
    window = pyglet.window.Window(width=BOARD_WIDTH*BLOCK_WIDTH,
                              height=BOARD_HEIGHT*BLOCK_HEIGHT)
else:
    window = None

board = Board(BOARD_WIDTH, BOARD_HEIGHT, block)
if PLOT:
    plot = Scoreplot()
    #plot.newscore(0,0)

game = RLGame(window, board, 1)
learner = QLearner(board, game, learningrate=0.1, epsilon=0)
#learner = SarsaLambdaLearner(board, game, learningrate=0.1, epsilon=0.0, lam=0.99)

# Load an existing policy if available
files = glob.glob('policy-*.pickle')
if files and len(files) > 0:
    print('Loading policy from file: '+files[-1])
    learner.policy = pickle.load(open(files[-1], 'r'))

if DRAW:
    @window.event
    def on_draw():
        if DRAW:
            game.draw_handler()

    @window.event
    def on_text_motion(motion):
        game.keyboard_handler(motion)

    @window.event
    def on_key_press(key_pressed, mod):
        if key_pressed == key.P:
            game.toggle_pause()


def update(dt):
    # do some serious learning here
    action = learner.step()
    # perform the action on the board
    board.move_piece(action)

    # update game
    game.cycle()

    # collect statistics
    global lastgame
    global lastscores

    if game.lines > 0 and game.lines != lastscores[-1]:
        # announce new piece to learner
        learner.newpiece()

    if game.lines > 0:
        if game.lines != lastscores[-1]:
            lastscores[-1] = game.lines

            maxscorebound = 1000
            if lastscores[-1] >= maxscorebound:
                # reset game
                print('Scored %d in game %d. Starting next game.'%(lastscores[-1], lastgame))
                game.manualreset()
                board.reset()
                learner.reset()

    if game._gamecounter > lastgame:
        lastgame = game._gamecounter
        if game._gamecounter % 100 == 0:
            avgscore = sum(lastscores)/float(len(lastscores))
            if PLOT:
                plot.newscore(lastgame, avgscore)
                plot.plot()

            print('Game: %d, Min score: %d, Max score: %d, Avg score: %f'%(game._gamecounter, min(lastscores), max(lastscores), avgscore))
            #print('Track length: %d'%len(learner.track))
            lastscores = []
        lastscores.append(0)
    # plot.plot()

try:
    if DRAW:
        pyglet.clock.schedule_interval(update, UPDATE_INTERVAL)
        pyglet.app.run()
    else:
        while 1:
            update(0)
except KeyboardInterrupt:
    print('Interrupted')
    # save policy
    filename = 'policy-%d.pickle'%game._gamecounter
    pickle.dump(learner.policy, open(filename, 'w'), -1)
    print('Policy saved to: '+filename)
