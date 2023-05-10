# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
from keyboardAgents import KeyboardAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  debug_choices = {'Stop': 0, 'North': 0, 'West': 0, 'East': 0, 'South': 0}
  numEnemyCapsules = None
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.numEnemyCapsules = len(self.getCapsules(gameState))

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    numCarrying = gameState.data.agentStates[self.index].numCarrying
    print(numCarrying)
    actions = gameState.getLegalActions(self.index)
    actions.remove("Stop")  # stopping just seems useless
    # You can profile your evaluation time by uncommenting these lines
    start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    print('eval time for agent %d: %.4f' % (self.index, time.time() - start), f'enemySide: {self.on_enemy_side(gameState)}')
    print(actions)
    print(values)
    #  print(self.debug_choices)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    print(bestActions)
    foodLeft = len(self.getFood(gameState).asList())
    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def on_enemy_side(self, gameState):
    if self.red:
      if gameState.data.food.width / 2 < gameState.getAgentState(self.index).getPosition()[0]:
        return True
    else:
      if gameState.data.food.width / 2 > gameState.getAgentState(self.index).getPosition()[0]:
        return True
    return False


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  RUNNING = False
  EVADING = False
  CAUTIOUS = False
  SAFE = True
  SUPER_SAFE = False
  capsuleTimer = 0
  carry_threshold = 6
  caution_distance = 7
  safe_distance = 10
  panic_distance = 6

  def collect_cautiously_feat(self, features, caution_increase, enemy_distance, food_distance, successor):
    print("Wary...")
    features['distanceToFood'] = food_distance
    features['distanceToEnemy'] = enemy_distance * caution_increase
    acts = successor.getLegalActions(self.index)
    futureChoices = len(acts)  # this will help us favor paths which have more choices available to them
    features['choicesAvailable'] = futureChoices
    return features

  def collect_freely_feat(self, features, food_list, minDistance):
    features['distanceToFood'] = minDistance
    features['successorScore'] = -len(food_list)  # self.getScore(successor)
    features['distanceToEnemy'] = 0
    return features

  def collect_capsule(self, feat, successor, capsule_distance, min_distance):
    print("EVADING!")
    acts = successor.getLegalActions(self.index)
    futureChoices = len(acts)  # this will help us favor paths which have more choices available to them
    # self.debug_choices[action] = futureChoices  # this is for debug
    feat['successorScore'] = -capsule_distance
    feat['choicesAvailable'] = futureChoices
    feat['distanceToEnemy'] = min_distance
    return feat

  def return_to_base_feat(self, feat, gameState, myPos, successor, min_distance, action):
    print("RUNNING!")
    acts = successor.getLegalActions(self.index)
    futureChoices = len(acts)  # this will help us favor paths which have more choices available to them
    self.debug_choices[action] = futureChoices
    if self.red:
      feat['successorScore'] = -abs((gameState.data.food.width / 2 - 10) - myPos[0])
      feat['distanceToEnemy'] = min_distance
      feat['choicesAvailable'] = futureChoices
    else:
      feat['successorScore'] = -abs((gameState.data.food.width / 2 + 10) - myPos[0])
      feat['distanceToEnemy'] = min_distance
      feat['choicesAvailable'] = futureChoices
    return feat

  def getFeatures(self, gameState, action):
    # initial setup for feature functions
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    food_list = self.getFood(successor).asList()
    enemy_list = self.getOpponents(successor)
    myPos = successor.getAgentState(self.index).getPosition()
    minDistance = None
    tempCapsuleNum = len(self.getCapsules(gameState))  # this will help us keep track of how many capsules we have, so we know when one is captured
    min_distance = min([self.getMazeDistance(myPos, successor.getAgentState(enemy).getPosition()) for enemy in enemy_list])
    features['distanceToEnemy'] = min_distance
    if len(food_list) > 0:  # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in food_list])
      numCarrying = gameState.data.agentStates[self.index].numCarrying
    # Compute distance to the nearest enemy
    if self.SUPER_SAFE:
      self.capsuleTimer -= 1
      if self.capsuleTimer <= 0:
        self.SUPER_SAFE = False
        self.SAFE = True
      if self.capsuleTimer < 7:
        if tempCapsuleNum >= 1:
          capsule_distance = min([self.getMazeDistance(myPos, capsule) for capsule in self.getCapsules(gameState)])
          return self.collect_capsule(features, successor, capsule_distance, min_distance)
        else:
          return self.return_to_base_feat(features, gameState, myPos, successor, min_distance, action)
      else:
        features['distanceToEnemy'] = 0
        return self.collect_freely_feat(features, food_list, minDistance)

    if numCarrying > self.carry_threshold:  # we check to see if we have enough food to return
      print(numCarrying)
      self.RUNNING = True
      self.CAUTIOUS = False
      self.SAFE = False
      self.EVADING = False
      return self.return_to_base_feat(features, gameState, myPos, successor, min_distance, action)
    if self.on_enemy_side(gameState):
      # features['distanceToEnemy'] = min_distance
      if min_distance > self.safe_distance:
        self.RUNNING = False
        self.CAUTIOUS = False
        self.SAFE = True
        self.EVADING = False
      elif min_distance < self.panic_distance:  # Book it
        self.EVADING = True
        self.RUNNING = False
        self.CAUTIOUS = False
        self.SAFE = False
      if self.RUNNING:
        return self.return_to_base_feat(features, gameState, myPos, successor, min_distance, action)
      elif self.SAFE:
        return self.collect_freely_feat(features, food_list, minDistance)
      elif self.EVADING:
        if self.numEnemyCapsules > tempCapsuleNum:
          self.numEnemyCapsules = tempCapsuleNum
          self.capsuleTimer = 40
          self.RUNNING = False
          self.EVADING = False
          self.CAUTIOUS = False
          self.SAFE = False
          self.SUPER_SAFE = True
          return self.collect_freely_feat(features, food_list, minDistance)
        elif tempCapsuleNum == 0:
          self.RUNNING = True
          self.EVADING = False
          self.CAUTIOUS = False
          self.SAFE = False
          self.SUPER_SAFE = False
          return self.return_to_base_feat(features, gameState, myPos, successor, min_distance, action)
        else:
          capsule_distance = min([self.getMazeDistance(myPos, capsule) for capsule in self.getCapsules(gameState)])
          return self.collect_capsule(features, successor, capsule_distance, min_distance)
    else:
      self.RUNNING = False
      self.EVADING = False
      self.CAUTIOUS = False
      self.SAFE = True
      features['distanceToEnemy'] = 0
      return self.collect_freely_feat(features, food_list, minDistance)

  def getWeights(self, gameState, action):
    if self.EVADING:
      return {'successorScore': 50, 'choicesAvailable': 70, 'distanceToEnemy': 70}
    elif self.RUNNING:
      return {'successorScore': 50, 'distanceToFood': -1, 'choicesAvailable': 60, 'distanceToEnemy': 60}
    elif self.CAUTIOUS:
      return {'successorScore': 80, 'distanceToFood': -1, 'choicesAvailable': 60, 'distanceToEnemy': 20}
    elif self.SAFE:
      return {'successorScore': 100, 'distanceToFood': -2, 'distanceToEnemy': 1}
    elif self.SUPER_SAFE:
      return {'successorScore': 100, 'distanceToFood': -2, 'choicesAvailable': 60, 'distanceToEnemy': 20}
    else:
      print("BUGBUGBUGBUGBUG")
      return {'successorScore': 50, 'distanceToFood': -1, 'choicesAvailable': 60, 'distanceToEnemy': 60}


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
