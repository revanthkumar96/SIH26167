#!/usr/bin/env bash
# Cost guardrails, created before the first training run rather than after the
# first surprise.
#
# Two thresholds because they mean different things: $40 is "review what is
# running", $75 is "the programme is at risk and something must stop". The
# grant is $100 and the plan spends about half of it.
#
# Budgets alarms are a warning, not a control -- they cannot stop an instance.
# The structural protection is infra/idle-shutdown.sh.
set -euo pipefail

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
EMAIL=${SATQUERY_ALERT_EMAIL:?set SATQUERY_ALERT_EMAIL to the address to notify}
NAME=${SATQUERY_BUDGET_NAME:-satquery-monthly}

create_budget() {
    local threshold=$1 label=$2
    aws budgets create-budget \
        --account-id "$ACCOUNT" \
        --budget "{
            \"BudgetName\": \"${NAME}-${label}\",
            \"BudgetLimit\": {\"Amount\": \"100\", \"Unit\": \"USD\"},
            \"TimeUnit\": \"MONTHLY\",
            \"BudgetType\": \"COST\"
        }" \
        --notifications-with-subscribers "[{
            \"Notification\": {
                \"NotificationType\": \"ACTUAL\",
                \"ComparisonOperator\": \"GREATER_THAN\",
                \"Threshold\": ${threshold},
                \"ThresholdType\": \"ABSOLUTE_VALUE\"
            },
            \"Subscribers\": [{
                \"SubscriptionType\": \"EMAIL\",
                \"Address\": \"${EMAIL}\"
            }]
        }]" 2>&1 | grep -v DuplicateRecordException || true
    echo "budget ${NAME}-${label} at \$${threshold}"
}

create_budget 40 review
create_budget 75 critical

echo
echo "existing budgets:"
aws budgets describe-budgets --account-id "$ACCOUNT" \
    --query 'Budgets[].[BudgetName,BudgetLimit.Amount]' --output table
