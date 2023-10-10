board = {
    "1": " ",
    "2": " ",
    "3": " ",
    "4": " ",
    "5": " ",
    "6": " ",
    "7": " ",
    "8": " ",
    "9": " ",
}
board_keys = list()
for k in board:
    board_keys.append(k)


# 화면 출력 함수 정의
def visual_board(board_num):
    print(board_num["1"], "|", board_num["2"], "|", board_num["3"], sep="")
    print(board_num["4"], "|", board_num["5"], "|", board_num["6"], sep="")
    print(board_num["7"], "|", board_num["8"], "|", board_num["9"], sep="")


# 보드 이동 함수 정의
def game():
    turn = "X"
    count = 0
    for i in range(8):
        visual_board(board)
        print("당신 차례입니다," + turn + ". 어디로 이동할까요?")
        move = input()
        if board[move] == " ":
            board[move] = turn
            count += 1
        else:
            print("이미 채워져 있습니다.")
            continue

        if count > 4:
            if board["1"] == board["2"] == board["3"] != " ":
                print_winner(turn)
                break
            if board["4"] == board["5"] == board["6"] != " ":
                print_winner(turn)
                break
            if board["7"] == board["8"] == board["9"] != " ":
                print_winner(turn)
                break
            if board["2"] == board["5"] == board["8"] != " ":
                print_winner(turn)
                break
            if board["3"] == board["6"] == board["9"] != " ":
                print_winner(turn)
                break
            if board["1"] == board["5"] == board["9"] != " ":
                print_winner(turn)
                break
            if board["3"] == board["5"] == board["7"] != " ":
                print_winner(turn)
                break
        if count == 9:
            print("\n게임 종료\n")
            print("동점입니다")

        if turn == "X":
            turn = "Y"
        else:
            turn = "X"


def print_winner(turn):
    visual_board(board)
    print("\n게임 종료\n")
    print("-" * 10, turn, "가 승리했습니다", "-" * 10)


if __name__ == "__main__":
    game()
