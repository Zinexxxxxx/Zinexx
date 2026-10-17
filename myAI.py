import random as r
import traceback
from collections import Counter

from poker.action import Action
from poker.game_state import GameState
from poker.hand import RANK_VALUES, evaluate
from poker.player import Blind

DEBUG = False  # set True to see errors in the terminal (the engine otherwise hides them)

last_bb = 2  # best guess at the current big blind, remembered between calls


def myAI(state: GameState) -> Action:
    try:
        return decide(state)
    except Exception:
        if DEBUG:
            traceback.print_exc()
        return defaultAI(state)


def defaultAI(state: GameState) -> Action:  # runs if myAI crashes, returns an invalid action or takes too long
    return Action.check() if state.my_player.chips_bet == max(player.chips_bet for player in state.players) else Action.fold()


# ---------------------------------------------------------------- helpers

def update_big_blind(state: GameState):
    # Only trust the blind amounts before anyone has raised, when the big blind is about double the small blind
    global last_bb
    big = next((p for p in state.players if p.blind == Blind.BIG), None)
    small = next((p for p in state.players if p.blind == Blind.SMALL), None)

    if big and small and small.chips_bet > 0 and small.chips_bet <= big.chips_bet <= 2 * small.chips_bet + 1:
        last_bb = big.chips_bet


def preflop_tier(a, b) -> int:
    # 3 = premium, 2 = good, 1 = playable, 0 = weak
    hi, lo = sorted((RANK_VALUES[a.rank], RANK_VALUES[b.rank]), reverse=True)
    suited = a.suit == b.suit

    if hi == lo:
        return 3 if hi >= 10 else 2 if hi >= 7 else 1

    if hi == 14 and lo >= 13:
        return 3
    if hi == 14 and lo >= 10:
        return 2
    if hi == 14 and suited:
        return 1
    if hi >= 13 and lo >= 11:
        return 2 if suited else 1
    if lo >= 10:
        return 1
    if suited and hi - lo <= 2 and lo >= 5:
        return 1

    return 0


def board_category(board) -> int:
    # The best hand the board makes on its own (so we can tell if our cards actually help)
    if len(board) >= 5:
        return evaluate(board)[0]

    counts = list(Counter(card.rank for card in board).values())

    if 4 in counts:
        return 7
    if 3 in counts:
        return 3
    if counts.count(2) >= 2:
        return 2
    if 2 in counts:
        return 1
    return 0


def draw_outs(hole, board) -> int:
    # Rough number of cards that would complete a flush or straight using at least one of our hole cards
    cards = hole + board
    outs = 0

    suits = Counter(card.suit for card in cards)
    for suit, count in suits.items():
        if count == 4 and any(card.suit == suit for card in hole):
            outs += 9

    values = {RANK_VALUES[card.rank] for card in cards}
    hole_values = {RANK_VALUES[card.rank] for card in hole}
    if 14 in values:
        values.add(1)
    if 14 in hole_values:
        hole_values.add(1)

    missing = set()
    for low in range(1, 11):
        window = set(range(low, low + 5))
        have = window & values
        if len(have) == 4 and have & hole_values:
            missing.add(next(iter(window - values)))
    outs += 4 * len(missing)

    return min(outs, 15)


# ---------------------------------------------------------------- the decision

def decide(state: GameState) -> Action:
    global last_bb

    me = state.my_player
    hole = list(me.hole_cards)
    board = list(state.community_cards)
    pot = state.chips_in_pot
    stack = me.chips_in_stack
    to_call = max(p.chips_bet for p in state.players) - me.chips_bet
    opponents = sum(1 for p in state.players if not p.has_folded and p is not me)

    if not board:
        update_big_blind(state)

    def call_or_check() -> Action:
        return Action.check() if to_call == 0 else Action.call()

    def check_or_fold() -> Action:
        return Action.check() if to_call == 0 else Action.fold()

    def raise_to(amount) -> Action:
        # raise_by is on top of the call, must be a whole number above 0, and can't be more than we have left
        amount = min(int(amount), stack - to_call)
        return Action.raise_by(amount) if amount > 0 else call_or_check()

    if not board:
        return preflop(hole, to_call, pot, stack, opponents, last_bb, call_or_check, check_or_fold, raise_to)

    return postflop(hole, board, to_call, pot, stack, opponents, call_or_check, check_or_fold, raise_to)


def preflop(hole, to_call, pot, stack, opponents, bb, call_or_check, check_or_fold, raise_to) -> Action:
    a, b = hole
    tier = preflop_tier(a, b)
    has_ace = 14 in (RANK_VALUES[a.rank], RANK_VALUES[b.rank])
    heads_up = opponents == 1

    # Short stack: push or fold, because blinds and antes eat you alive otherwise
    if stack < 10 * bb:
        if tier >= 1 or has_ace:
            return raise_to(stack)
        return check_or_fold()

    if tier == 3:
        return raise_to(max(bb * 3, pot * 0.75))

    if tier == 2:
        if to_call <= bb * 3:
            return raise_to(max(bb * 2, pot * 0.5))
        return call_or_check() if to_call <= stack * 0.25 else Action.fold()

    if tier == 1:
        if heads_up and to_call <= bb:
            return raise_to(bb * 2)
        return call_or_check() if to_call <= max(bb * 2, stack * 0.05) else Action.fold()

    # Weak hand: sometimes steal the blinds when everyone else has folded
    if heads_up and to_call <= bb and r.random() < 0.35:
        return raise_to(bb * 2.5)

    return check_or_fold()


def postflop(hole, board, to_call, pot, stack, opponents, call_or_check, check_or_fold, raise_to) -> Action:
    cards = hole + board
    score = evaluate(cards)
    category = score[0]
    made = category > board_category(board)  # our cards really help, not just the board

    board_high = max(RANK_VALUES[card.rank] for card in board)
    outs = draw_outs(hole, board) if len(board) < 5 else 0
    draw_equity = outs * (0.04 if len(board) == 3 else 0.02)

    # Work out roughly how often we're winning
    if made and category >= 3:
        tier, equity = "monster", 0.92
    elif made and category == 2:
        tier, equity = "strong", 0.75
    elif made and category == 1:
        tier, equity = ("top pair", 0.6) if score[1] >= board_high else ("weak pair", 0.33)
    else:
        tier, equity = "air", 0.03

    equity = max(equity, draw_equity)
    if tier in ("top pair", "weak pair"):
        equity *= 0.92 ** max(0, opponents - 1)  # more opponents, more chance someone beats a pair

    heads_up = opponents == 1

    # Nobody has bet: bet good hands, semi-bluff draws, and bluff sometimes when heads-up
    if to_call == 0:
        if tier in ("monster", "strong"):
            return raise_to(pot * r.uniform(0.6, 0.8))
        if tier == "top pair":
            return raise_to(pot * 0.5)
        if outs >= 8 and r.random() < 0.5:
            return raise_to(pot * 0.5)
        if tier == "weak pair" and heads_up and r.random() < 0.3:
            return raise_to(pot * 0.4)
        if tier == "air" and heads_up and r.random() < 0.3:
            return raise_to(pot * 0.5)
        return Action.check()

    # Facing a bet: use pot odds
    needed = to_call / (pot + to_call)

    if tier == "monster":
        return raise_to(pot * 0.75) if r.random() < 0.7 else call_or_check()

    if tier == "strong":
        if to_call <= pot * 0.5 and r.random() < 0.5:
            return raise_to(pot * 0.6)
        return call_or_check()

    if outs >= 8 and heads_up and r.random() < 0.15:
        return raise_to(pot * 0.6)  # semi-bluff

    if equity >= needed + 0.02:
        return call_or_check()

    return Action.fold()
