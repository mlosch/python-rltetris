import pyglet
from pyglet.window import key
from lib.Board import Board
from lib.Game import RLGame
from lib.Learning import QLearner, SarsaLambdaLearner
from lib.util import Scoreplot
import cPickle as pickle
import glob


"""
Things to play around with:
Checkout the following files for further adjustable parameters: Shape.py, Game.py, Learning.py

Change DRAW to True, to see the agent play.
Change the UPDATE_INTERVAL (in seconds) to define how many moves the agent can make per second.
Change PLOT to True, to draw a plot showing the average reward over games.

Change BOARD_WIDTH and BOARD_HEIGHT to define the size of the game board.

Change learner (in the middle of this file) to define which algorithm is used ( currently either Q or SARSA(lambda) )
"""
DRAW = False
UPDATE_INTERVAL = 0.1
PLOT = False

BOARD_WIDTH = 6
BOARD_HEIGHT = 12


global lastgame
global lastscores
lastgame = 0
lastscores = [0]

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

game = RLGame(window, board, 1)
#learner = SarsaLearner(board, game, learningrate=0.1, epsilon=0.0, discountfactor=0.7)
#learner = QLearner(board, game, learningrate=0.1, epsilon=0.05, discountfactor=0.7)
learner = SarsaLambdaLearner(board, game, learningrate=0.01, epsilon=0.0, lam=0.9, discountfactor=0.7)

# Load an existing policy if available
files = glob.glob('policy-*.pickle')
if files and len(files) > 0:
    print('Loading policy from file: '+files[-1])
    file = files[-1]
    file = file[file.find('-')+1:file.find('.')]
    policy_iter = int(file)
    learner.policy = pickle.load(open(files[-1], 'r'))
    game._gamecounter = policy_iter

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
