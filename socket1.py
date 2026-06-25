import socket
import threading
import state

def start_input_server(host='0.0.0.0', port=8888):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[SERVER] Đang chờ kết nối tại {host}:{port}...")

    conn, addr = server.accept()
    print(f"[SERVER] Đã kết nối với {addr}")
    with conn:
        while True:
            try:
                data = conn.recv(1024).decode().strip()
                if not data:
                    break

                print(f"[SERVER] Nhận: {data}")
                parts = data.split(',')
                if len(parts) >= 2:
                    b1 = int(parts[0])
                    b2 = int(parts[1])
                    if b1 == 1 and b2 == 0:
                        state.running = not state.running
                        print(f"[STATE] running = {state.running}")
                    elif b1 == 0 and b2 == 1:
                        state.running_inference = not state.running_inference
                        print(f"[STATE] running_inference = {state.running_inference}")
                    elif b1 == 1 and b2 == 1 and len(parts) == 3:
                        opt = int(parts[2])
                        state.count_alert_AI = 2 ** opt
                        print(f"[STATE] count_alert_AI = {state.count_alert_AI}")
            except Exception as e:
                print(f"[SERVER] Lỗi: {e}")
                break

    server.close()

if __name__ == "__main__":
    start_input_server()