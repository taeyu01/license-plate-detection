#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#define PWM_PATH "/sys/class/pwm/pwmchip0/pwm0"

void write_pwm(const char *file, const char *value)
{
    char path[256];

    snprintf(path, sizeof(path), "%s/%s", PWM_PATH, file);

    int fd = open(path, O_WRONLY);

    if (fd < 0)
    {
        perror("open");
        exit(1);
    }

    if (write(fd, value, strlen(value)) < 0)
    {
        perror("write");
        close(fd);
        exit(1);
    }

    close(fd);
}

int main()
{
    // 50 Hz = 20 ms
    write_pwm("period", "20000000");

    // 중앙 위치 ≈ 1.5 ms
    write_pwm("duty_cycle", "1500000");

    // PWM 출력 시작
    write_pwm("enable", "1");

    printf("SG90 -> CENTER\n");

    sleep(2);

    // PWM 출력 종료
    write_pwm("enable", "0");

    return 0;
}