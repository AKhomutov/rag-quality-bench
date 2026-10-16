# Acme Cloud Support Policy

## Support Availability

Acme Cloud provides standard customer support from Monday through Friday, between 08:00 and 18:00 UTC.

Standard support is not available on weekends or on public holidays observed by the support team.

Enterprise customers with the Critical Support add-on have access to emergency support 24 hours a day, 7 days a week.

## Incident Severity Levels

Acme Cloud classifies incidents into four severity levels.

### Severity 1 — Critical

A Severity 1 incident means that the production service is completely unavailable for all users, or that a confirmed security incident is actively exposing customer data.

The initial response target for Severity 1 incidents is 30 minutes.

### Severity 2 — High

A Severity 2 incident means that a major production feature is unavailable for a significant number of users, but the service as a whole is still operational.

The initial response target for Severity 2 incidents is 2 hours.

### Severity 3 — Medium

A Severity 3 incident means that a non-critical feature is malfunctioning and a workaround is available.

The initial response target for Severity 3 incidents is 8 business hours.

### Severity 4 — Low

A Severity 4 incident includes cosmetic defects, minor usability issues, and general product questions.

The initial response target for Severity 4 incidents is 2 business days.

## Scheduled Maintenance

Routine production maintenance is normally performed on Sundays between 02:00 and 04:00 UTC.

Customers must be notified at least 72 hours before scheduled maintenance that is expected to cause more than 10 minutes of service interruption.

Emergency maintenance may be performed without 72 hours of advance notice when delaying the maintenance would create a significant security or availability risk.

## Data Retention

Customer application logs are retained for 30 days by default.

Enterprise customers may configure application log retention for up to 365 days.

Deleted customer workspaces remain recoverable for 14 days after deletion. After the recovery period expires, workspace data is permanently deleted from active systems.

Backups containing deleted workspace data may persist for up to 30 additional days before automatic expiration.

## Refund Policy

Monthly subscriptions are refundable only when the refund request is submitted within 7 calendar days of the initial purchase.

Subscription renewals are not eligible for refunds.

Annual subscriptions may receive a prorated refund only when Acme Cloud terminates the service without cause.

Service outages do not automatically qualify customers for refunds. Eligible customers may instead receive service credits under the Service Level Agreement.

## Service Level Agreement

The monthly uptime target for the Acme Cloud production platform is 99.9%.

Scheduled maintenance announced according to the maintenance policy is excluded from the uptime calculation.

If monthly uptime falls below 99.9%, eligible customers may request service credits.

Service credits must be requested within 30 days after the end of the affected calendar month.

## Account Security

Multi-factor authentication is optional for individual accounts.

Multi-factor authentication is mandatory for organization administrators on Enterprise plans.

Acme Cloud employees will never ask customers to provide their password, authentication code, recovery code, or API secret through email or support chat.

## API Rate Limits

The standard API rate limit is 600 requests per minute per workspace.

Enterprise customers may request a custom rate limit.

Requests that exceed the configured limit return HTTP status code 429.

Rate-limit counters reset continuously and are not based on fixed clock-minute boundaries.

## Data Export

Workspace administrators may request a complete export of workspace data.

Exports
