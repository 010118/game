"""
gtp_connection.py
Module for playing games of Go using GoTextProtocol

Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller.
Parts of this code were originally based on the gtp module 
in the Deep-Go project by Isaac Henrion and Amos Storkey 
at the University of Edinburgh.
"""
import traceback
import numpy as np
import re
import random
from sys import stdin, stdout, stderr
from typing import Any, Callable, Dict, List, Tuple


from board_base import (
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    GO_COLOR,
    GO_POINT,
    PASS,
    MAXSIZE,
    coord_to_point,
    opponent,
)
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine


class GtpConnection:
    def __init__(
        self, go_engine: GoEngine, board: GoBoard, debug_mode: bool = False
    ) -> None:
        """
        Manage a GTP connection for a Go-playing engine

        Parameters
        ----------
        go_engine:
            a program that can reply to a set of GTP commandsbelow
        board:
            Represents the current board state.
        """
        # self.BW_count = [0,0]
        self.B = 0
        self.W = 0
        self._debug_mode: bool = debug_mode
        self.go_engine = go_engine
        self.board: GoBoard = board
        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "protocol_version": self.protocol_version_cmd,
            "quit": self.quit_cmd,
            "name": self.name_cmd,
            "boardsize": self.boardsize_cmd,
            "showboard": self.showboard_cmd,
            "clear_board": self.clear_board_cmd,
            "komi": self.komi_cmd,
            "version": self.version_cmd,
            "known_command": self.known_command_cmd,
            "genmove": self.genmove_cmd,
            "list_commands": self.list_commands_cmd,
            "play": self.play_cmd,
            "legal_moves": self.legal_moves_cmd,
            "gogui-rules_legal_moves": self.gogui_rules_legal_moves_cmd,
            "gogui-rules_final_result": self.gogui_rules_final_result_cmd,
            "gogui-rules_captured_count": self.gogui_rules_captured_count_cmd,
            "gogui-rules_game_id": self.gogui_rules_game_id_cmd,
            "gogui-rules_board_size": self.gogui_rules_board_size_cmd,
            "gogui-rules_side_to_move": self.gogui_rules_side_to_move_cmd,
            "gogui-rules_board": self.gogui_rules_board_cmd,
            "gogui-analyze_commands": self.gogui_analyze_cmd,
        }

        # argmap is used for argument checking
        # values: (required number of arguments,
        #          error message on argnum failure)
        self.argmap: Dict[str, Tuple[int, str]] = {
            "boardsize": (1, "Usage: boardsize INT"),
            "komi": (1, "Usage: komi FLOAT"),
            "known_command": (1, "Usage: known_command CMD_NAME"),
            "genmove": (1, "Usage: genmove {w,b}"),
            "play": (2, "Usage: play {b,w} MOVE"),
            "legal_moves": (1, "Usage: legal_moves {w,b}"),
        }

    def write(self, data: str) -> None:
        stdout.write(data)

    def flush(self) -> None:
        stdout.flush()

    def start_connection(self) -> None:
        """
        Start a GTP connection.
        This function continuously monitors standard input for commands.
        """
        line = stdin.readline()
        while line:
            self.get_cmd(line)
            line = stdin.readline()

    def get_cmd(self, command: str) -> None:
        """
        Parse command string and execute it
        """
        if len(command.strip(" \r\t")) == 0:
            return
        if command[0] == "#":
            return
        # Strip leading numbers from regression tests
        if command[0].isdigit():
            command = re.sub("^\d+", "", command).lstrip()

        elements: List[str] = command.split()
        if not elements:
            return
        command_name: str = elements[0]
        args: List[str] = elements[1:]
        if self.has_arg_error(command_name, len(args)):
            return
        if command_name in self.commands:
            try:
                self.commands[command_name](args)
            except Exception as e:
                self.debug_msg("Error executing command {}\n".format(str(e)))
                self.debug_msg("Stack Trace:\n{}\n".format(traceback.format_exc()))
                raise e
        else:
            self.debug_msg("Unknown command: {}\n".format(command_name))
            self.error("Unknown command")
            stdout.flush()

    def has_arg_error(self, cmd: str, argnum: int) -> bool:
        """
        Verify the number of arguments of cmd.
        argnum is the number of parsed arguments
        """
        if cmd in self.argmap and self.argmap[cmd][0] != argnum:
            self.error(self.argmap[cmd][1])
            return True
        return False

    def debug_msg(self, msg: str) -> None:
        """Write msg to the debug stream"""
        if self._debug_mode:
            stderr.write(msg)
            stderr.flush()

    def error(self, error_msg: str) -> None:
        """Send error msg to stdout"""
        stdout.write("? {}\n\n".format(error_msg))
        stdout.flush()

    def respond(self, response: str = "") -> None:
        """Send response to stdout"""
        stdout.write("= {}\n\n".format(response))
        stdout.flush()

    def reset(self, size: int) -> None:
        """
        Reset the board to empty board of given size
        """
        self.board.reset(size)

    def board2d(self) -> str:
        return str(GoBoardUtil.get_twoD_board(self.board))

    def protocol_version_cmd(self, args: List[str]) -> None:
        """Return the GTP protocol version being used (always 2)"""
        self.respond("2")

    def quit_cmd(self, args: List[str]) -> None:
        """Quit game and exit the GTP interface"""
        self.respond()
        exit()

    def name_cmd(self, args: List[str]) -> None:
        """Return the name of the Go engine"""
        self.respond(self.go_engine.name)

    def version_cmd(self, args: List[str]) -> None:
        """Return the version of the  Go engine"""
        self.respond(str(self.go_engine.version))

    def clear_board_cmd(self, args: List[str]) -> None:
        """clear the board"""
        # self.BW_count = [0, 0]
        self.B = 0
        self.W = 0
        self.reset(self.board.size)
        self.respond()

    def boardsize_cmd(self, args: List[str]) -> None:
        """
        Reset the game with new boardsize args[0]
        """
        self.reset(int(args[0]))
        self.respond()

    def showboard_cmd(self, args: List[str]) -> None:
        self.respond("\n" + self.board2d())

    def komi_cmd(self, args: List[str]) -> None:
        """
        Set the engine's komi to args[0]
        """
        self.go_engine.komi = float(args[0])
        self.respond()

    def known_command_cmd(self, args: List[str]) -> None:
        """
        Check if command args[0] is known to the GTP interface
        """
        if args[0] in self.commands:
            self.respond("true")
        else:
            self.respond("false")

    def list_commands_cmd(self, args: List[str]) -> None:
        """list all supported GTP commands"""
        self.respond(" ".join(list(self.commands.keys())))

    def legal_moves_cmd(self, args: List[str]) -> None:
        """
        List legal moves for color args[0] in {'b','w'}
        """
        board_color: str = args[0].lower()
        color: GO_COLOR = color_to_int(board_color)
        moves: List[GO_POINT] = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves: List[str] = []
        for move in moves:
            coords: Tuple[int, int] = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = " ".join(sorted(gtp_moves))
        self.respond(sorted_moves)

    """
    ==========================================================================
    Assignment 1 - game-specific commands start here
    ==========================================================================
    """
    """
    ==========================================================================
    Assignment 1 - commands we already implemented for you
    ==========================================================================
    """

    def gogui_analyze_cmd(self, args: List[str]) -> None:
        """We already implemented this function for Assignment 1"""
        self.respond(
            "pstring/Legal Moves For ToPlay/gogui-rules_legal_moves\n"
            "pstring/Side to Play/gogui-rules_side_to_move\n"
            "pstring/Final Result/gogui-rules_final_result\n"
            "pstring/Board Size/gogui-rules_board_size\n"
            "pstring/Rules GameID/gogui-rules_game_id\n"
            "pstring/Show Board/gogui-rules_board\n"
        )

    def gogui_rules_game_id_cmd(self, args: List[str]) -> None:
        """We already implemented this function for Assignment 1"""
        self.respond("Ninuki")

    def gogui_rules_board_size_cmd(self, args: List[str]) -> None:
        """We already implemented this function for Assignment 1"""
        self.respond(str(self.board.size))

    def gogui_rules_side_to_move_cmd(self, args: List[str]) -> None:
        """We already implemented this function for Assignment 1"""
        color = "black" if self.board.current_player == BLACK else "white"
        self.respond(color)

    def gogui_rules_board_cmd(self, args: List[str]) -> None:
        """We already implemented this function for Assignment 1"""
        size = self.board.size
        str = ""
        for row in range(size - 1, -1, -1):
            start = self.board.row_start(row + 1)
            for i in range(size):
                # str += '.'
                point = self.board.board[start + i]
                if point == BLACK:
                    str += "X"
                elif point == WHITE:
                    str += "O"
                elif point == EMPTY:
                    str += "."
                else:
                    assert False
            str += "\n"
        self.respond(str)

    """
    ==========================================================================
    Assignment 1 - game-specific commands you have to implement or modify
    ==========================================================================
    """

    def gogui_rules_final_result_cmd(self, args: List[str]) -> None:
        """Implement this function for Assignment 1"""
        draw = False
        found = False
        temp_board = GoBoardUtil.get_twoD_board(
            self.board
        )  # GoBoardUtil.get_twoD_board(self.board)
        # return a 2-dim array, 0 is row and 1 is col
        # print(GoBoardUtil.get_twoD_board(self.board))
        count = 0
        index = len(temp_board)
        for row in range(index):
            if count == index:
                draw = True
            if 0 not in temp_board[row]:
                count += 1
            if found == True:
                break
            for col in range(index):
                color = temp_board[row][col]
                if color != EMPTY:
                    if self.go_through_down(temp_board, row, col, color, index):
                        if color == 1:
                            self.respond("black")
                        else:
                            self.respond("white")
                        found = True
                        return 0
                    if self.go_through_up(temp_board, row, col, color, index):
                        if color == 1:
                            self.respond("black")
                        else:
                            self.respond("white")
                        found = True
                        return 0
                    if self.go_through_left(temp_board, row, col, color, index):
                        if color == 1:
                            self.respond("black")
                        else:
                            self.respond("white")
                        found = True
                        return 0
                    if self.go_through_right(temp_board, row, col, color, index):
                        if color == 1:
                            self.respond("black")
                        else:
                            self.respond("white")
                        found = True
                        return 0
                    if self.go_through_NW(temp_board, row, col, color, index):
                        if color == 1:
                            self.respond("black")
                        else:
                            self.respond("white")
                        found = True
                        return 0
                    if self.go_through_NE(temp_board, row, col, color, index):
                        if color == 1:
                            self.respond("black")
                        else:
                            self.respond("white")
                        found = True
                        return 0
                    if self.go_through_SE(temp_board, row, col, color, index):
                        if color == 1:
                            self.respond("black")
                        else:
                            self.respond("white")
                        found = True
                        return 0
                    if self.go_through_SW(temp_board, row, col, color, index):
                        if color == 1:
                            self.respond("black")
                        else:
                            self.respond("white")
                        found = True
                        return 0
        if draw == False:
            if self.B == 10:
                self.respond("white")
                return
            elif self.W == 10:
                self.respond("black")
                return
            else:
                self.respond("unknown")
                return

        if draw == True:
            self.respond("Draw")
            return None

    def temp_result(self):
        temp_board = GoBoardUtil.get_twoD_board(self.board)
        index = len(temp_board[0])

        break_outer_loop = False
        # CHECK whether there are 5 same color
        for row in range(index):
            if break_outer_loop == True:
                break
            for col in range(index):
                color = temp_board[row][col]
                if color != EMPTY:
                    if self.go_through_down(temp_board, row, col, color, index):
                        if color == 1:
                            return 1
                        else:
                            return 2
                    if self.go_through_up(temp_board, row, col, color, index):
                        if color == 1:
                            return 1
                        else:
                            return 2
                    if self.go_through_left(temp_board, row, col, color, index):
                        if color == 1:
                            return 1
                        else:
                            return 2
                    if self.go_through_right(temp_board, row, col, color, index):
                        if color == 1:
                            return 1
                        else:
                            return 2
                    if self.go_through_NW(temp_board, row, col, color, index):
                        if color == 1:
                            return 1
                        else:
                            return 2
                    if self.go_through_NE(temp_board, row, col, color, index):
                        if color == 1:
                            return 1
                        else:
                            return 2
                    if self.go_through_SE(temp_board, row, col, color, index):
                        if color == 1:
                            return 1
                        else:
                            return 2
                    if self.go_through_SW(temp_board, row, col, color, index):
                        if color == 1:
                            return 1
                        else:
                            return 2

    def gogui_rules_legal_moves_cmd(self, args: List[str]) -> None:
        """Implement this function for Assignment 1"""
        """
        If the game is over, return an empty list. Otherwise, 
        return a list of all empty points on the board in sorted order.
        In the starter code, this command always returns an empty list.
        """
        temp_board = GoBoardUtil.get_twoD_board(self.board)
        index = len(temp_board[0])
        win_color = None
        break_outer_loop = False
        # CHECK whether there are 5 same color
        if self.B == 10:
            self.respond()
            return
        elif self.W == 10:
            self.respond()
            return 
        for row in range(index):
            if break_outer_loop == True:
                break
            for col in range(index):
                color = temp_board[row][col]
                if color != EMPTY:
                    if self.go_through_down(temp_board, row, col, color, index):
                        self.respond()
                        return
                    if self.go_through_up(temp_board, row, col, color, index):
                        self.respond()
                        return
                    if self.go_through_left(temp_board, row, col, color, index):
                        self.respond()
                        return
                    if self.go_through_right(temp_board, row, col, color, index):
                        self.respond()
                        return
                    if self.go_through_NW(temp_board, row, col, color, index):
                        self.respond()
                        return
                    if self.go_through_NE(temp_board, row, col, color, index):
                        self.respond()
                        return
                    if self.go_through_SE(temp_board, row, col, color, index):
                        self.respond()
                        return
                    if self.go_through_SW(temp_board, row, col, color, index):
                        self.respond()
                        return
        legal_moves_list = self.legal_move()
        # sort in alphabetic order.
        sorted_moves = " ".join(sorted(legal_moves_list))
        self.respond(str(sorted_moves))
        

    def legal_move(self):
        current_play = self.board.current_player
        # return legal moves

        legal_moves = GoBoardUtil.generate_legal_moves(self.board, current_play)
        legal_moves_list = []

        for move in legal_moves:
            pos = point_to_coord(move, self.board.size)
            legal_moves_list.append(format_point(pos))
        return legal_moves_list

    def go_through_up(self, temp_board, row, col, target_color, index):
        temp_count = 1
        go_through = True
        while go_through == True and (row - 1) >= 0:
            if temp_board[row - 1][col] == target_color:
                temp_count += 1
                row -= 1
            else:
                return False
            # return if count 5 same stones
            if temp_count >= 5:
                go_through = False
                return True

            # return if it is out of range
            if row < 0:
                return False

    def go_through_down(self, temp_board, row, col, target_color, index):
        temp_count = 1
        go_through = True
        # need to check whether it is out of range on next step
        while go_through == True and (row + 1) < index:
            if temp_board[row + 1][col] == target_color:
                temp_count += 1
                row += 1
                go_through = True
            else:
                return False
            if temp_count >= 5:
                go_through = False
                return True

            if row >= index:
                return False

    def go_through_left(self, temp_board, row, col, target_color, index):
        temp_count = 1
        go_through = True
        while go_through == True and (col - 1) >= 0:
            if temp_board[row][col - 1] == target_color:
                temp_count += 1
                col -= 1
                go_through = True
            else:
                return False
            if temp_count >= 5:
                go_through = False
                return True

            if col < 0:
                return False

    def go_through_right(self, temp_board, row, col, target_color, index):
        temp_count = 1
        go_through = True
        while go_through == True and (col + 1) < index:
            if temp_board[row][col + 1] == target_color:
                temp_count += 1
                col += 1
                go_through = True
            else:
                return False
            if temp_count >= 5:
                go_through = False
                return True

            if col >= index:
                return False

    def go_through_NW(self, temp_board, row, col, target_color, index):
        temp_count = 1
        go_through = True
        while go_through == True and (row - 1) >= 0 and (col + 1) < index:
            if temp_board[row - 1][col + 1] == target_color:
                temp_count += 1
                col += 1
                row -= 1
                go_through = True
            else:
                return False
            if temp_count >= 5:
                go_through = False
                return True

            if row < 0:
                return False
            if col >= index:
                return False

    def go_through_NE(self, temp_board, row, col, target_color, index):
        temp_count = 1
        go_through = True
        while go_through == True and (row - 1) >= 0 and (col - 1) >= 0:
            if temp_board[row - 1][col - 1] == target_color:
                temp_count += 1
                col -= 1
                row -= 1
                go_through = True
            else:
                return False

            if temp_count >= 5:
                go_through = False
                return True

            if row < 0:
                return False
            if col < 0:
                return False

    def go_through_SW(self, temp_board, row, col, target_color, index):
        temp_count = 1
        go_through = True
        while go_through == True and (row + 1) < index and (col - 1) >= 0:
            if temp_board[row + 1][col - 1] == target_color:
                temp_count += 1
                col -= 1
                row += 1
                go_through = True
            else:
                return False

            if temp_count >= 5:
                go_through = False
                return True

            if row >= index:
                return False
            if col < 0:
                return False

    def go_through_SE(self, temp_board, row, col, target_color, index):
        temp_count = 1
        go_through = True
        while go_through == True and (row + 1) < index and (col + 1) < index:
            if temp_board[row + 1][col + 1] == target_color:
                temp_count += 1
                col += 1
                row += 1
                go_through = True
            else:
                return False
            if temp_count >= 5:
                go_through = False
                return True
            if row >= index:
                return False
            if col >= index:
                return False

    def play_cmd(self, args: List[str]) -> None:
        """
        Modify this function for Assignment 1.
        play a move args[1] for given color args[0] in {'b','w'}.
        """

        board_color = args[0].lower()
        board_move = args[1]

        error_list = ["wrong color", "wrong coordinate", "occupied"]
        try:
            # if not is_black_white(color):#return true if it is w or b, otherwise False
            # pass
            if color_to_int(board_color):
                color = color_to_int(board_color)

        except Exception as q:
            self.respond(f'Illegal Move: "{args}" {error_list[0]}')
            return
        # move is a go point(int)
        try:
            if move_to_coord(
                args[1], self.board.size
            ):  # move to coord can raise an error if cannot pass
                coord = move_to_coord(args[1], self.board.size)

        except Exception as w:
            temp = str(f'Illegal Move: "{args}" {error_list[1]}')
            self.respond(temp)
            return  # return Go-point

        legal_list = self.legal_move()
        # print(legal_list)
        if board_move.lower() == "pass":
            self.board.play_move(PASS, color)
            self.board.current_player = opponent(color)
            self.respond()
            return
        move = coord_to_point(coord[0], coord[1], self.board.size)
        if board_move.upper() in legal_list:
            catch_arr = self.board.capture_by_a1(move, color)
            if catch_arr[1] == BLACK:
                self.W += catch_arr[0]
            elif catch_arr[1] == WHITE:
                self.B += catch_arr[0]
            self.respond()
        else:
            temp = str("Illegal Move: {},{}".format(board_move, error_list[2]))
            self.respond(temp)
            return

    def genmove_cmd(self, args: List[str]) -> None:
        """
        Modify this function for Assignment 1.
        Generate a move for color args[0] in {'b','w'}.
        """
        legal_list = self.legal_move()
        board_color = args[0].lower()
        color = color_to_int(board_color)
        move_as_string = random.choice(legal_list)
        move_coord = move_to_coord(move_as_string, self.board.size)
        """
        move = self.go_engine.get_move(self.board, color)
        move_coord = point_to_coord(move, self.board.size)

        move_as_string = format_point(move_coord)
        """

        check_move = coord_to_point(move_coord[0], move_coord[1], self.board.size)
        if self.temp_result() == 1 or self.temp_result() == 2:
            self.respond("resign")
            return
        if len(legal_list) != 0:
            catch_arr = self.board.capture_by_a1(check_move, color)
            if catch_arr[1] == BLACK:
                self.W += catch_arr[0]
            elif catch_arr[1] == WHITE:
                self.B += catch_arr[0]
            self.respond(str(move_as_string))
        else:
            temp = "Illegal move: {}".format(move_as_string)
            self.respond(temp)

    def gogui_rules_captured_count_cmd(self, args: List[str]) -> None:
        """
        Modify this function for Assignment 1.
        Respond with the score for white, an space, and the score for black.
        """
        temp = str("{} {}".format(self.B, self.W))
        self.respond(temp)
        # self.respond("[" + " ".join(map(str, self.B,self.W)) + "]")

    """
    ==========================================================================
    Assignment 1 - game-specific commands end here
    ==========================================================================
    """


def point_to_coord(point: GO_POINT, boardsize: int) -> Tuple[int, int]:
    """
    Transform point given as board array index
    to (row, col) coordinate representation.
    Special case: PASS is transformed to (PASS,PASS)
    """
    if point == PASS:
        return (PASS, PASS)
    else:
        NS = boardsize + 1
        return divmod(point, NS)


def format_point(move: Tuple[int, int]) -> str:
    """
    Return move coordinates as a string such as 'A1', or 'PASS'.
    """
    assert MAXSIZE <= 25
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if move[0] == PASS:
        return "PASS"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1] + str(row)


def move_to_coord(point_str: str, board_size: int) -> Tuple[int, int]:
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    Raises ValueError if point_str is invalid
    """
    if not 2 <= board_size <= MAXSIZE:
        raise ValueError("board_size out of range")
    s = point_str.lower()
    if s == "pass":
        return (PASS, PASS)
    try:
        col_c = s[0]
        if (not "a" <= col_c <= "z") or col_c == "i":
            raise ValueError
        col = ord(col_c) - ord("a")
        if col_c < "i":
            col += 1
        row = int(s[1:])
        if row < 1:
            raise ValueError
    except (IndexError, ValueError):
        raise ValueError("invalid point: '{}'".format(s))
    if not (col <= board_size and row <= board_size):
        raise ValueError("point off board: '{}'".format(s))
    return row, col


def color_to_int(c: str) -> int:
    """convert character to the appropriate integer code"""
    color_to_int = {"b": BLACK, "w": WHITE, "e": EMPTY, "BORDER": BORDER}
    return color_to_int[c]
