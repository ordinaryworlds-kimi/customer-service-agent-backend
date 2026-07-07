-- mall_agent 独立库建表脚本
-- MySQL 5.7+ / 8.0，字符集 utf8mb4

CREATE DATABASE IF NOT EXISTS `mall_agent` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `mall_agent`;

-- 会话表
DROP TABLE IF EXISTS `conversation`;
CREATE TABLE `conversation` (
  `id`            BIGINT       NOT NULL AUTO_INCREMENT COMMENT '会话ID',
  `member_id`     BIGINT       NOT NULL COMMENT 'mall 会员ID',
  `member_username` VARCHAR(64) NOT NULL COMMENT '会员用户名',
  `title`         VARCHAR(200) DEFAULT NULL COMMENT '会话标题',
  `status`        TINYINT      NOT NULL DEFAULT 1 COMMENT '状态：0->关闭；1->进行中',
  `created_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_member_id` (`member_id`),
  KEY `idx_updated_at` (`updated_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';

-- 消息表
DROP TABLE IF EXISTS `message`;
CREATE TABLE `message` (
  `id`              BIGINT       NOT NULL AUTO_INCREMENT COMMENT '消息ID',
  `conversation_id` BIGINT       NOT NULL COMMENT '会话ID',
  `role`            VARCHAR(20)  NOT NULL COMMENT '角色：user/assistant/system/tool',
  `content`         TEXT         NOT NULL COMMENT '消息内容',
  `agent_name`      VARCHAR(50)  DEFAULT NULL COMMENT '处理的 Agent 名称',
  `token_usage`     INT          DEFAULT NULL COMMENT 'Token 消耗',
  `created_at`      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_conversation_id` (`conversation_id`),
  CONSTRAINT `fk_message_conversation` FOREIGN KEY (`conversation_id`) REFERENCES `conversation` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话消息';

-- 长期记忆表
DROP TABLE IF EXISTS `memory`;
CREATE TABLE `memory` (
  `id`          BIGINT       NOT NULL AUTO_INCREMENT COMMENT '记忆ID',
  `member_id`   BIGINT       NOT NULL COMMENT '会员ID',
  `memory_type` VARCHAR(50)  NOT NULL COMMENT '类型：preference/address/product/habit/other',
  `memory_key`  VARCHAR(100) NOT NULL COMMENT '记忆键',
  `memory_value` TEXT        NOT NULL COMMENT '记忆内容',
  `source`      VARCHAR(50)  DEFAULT 'agent' COMMENT '来源：agent/user/system',
  `created_at`  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at`  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_member_type_key` (`member_id`, `memory_type`, `memory_key`),
  KEY `idx_member_id` (`member_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户长期记忆';

-- 商品缓存表（RAG/Tool 加速）
DROP TABLE IF EXISTS `product_cache`;
CREATE TABLE `product_cache` (
  `id`           BIGINT       NOT NULL AUTO_INCREMENT,
  `product_id`   BIGINT       NOT NULL COMMENT 'mall 商品ID',
  `product_name` VARCHAR(200) NOT NULL,
  `summary`      TEXT         DEFAULT NULL COMMENT '商品摘要',
  `payload_json` JSON         DEFAULT NULL COMMENT '原始数据快照',
  `synced_at`    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_product_id` (`product_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品缓存';

-- Tool 调用日志
DROP TABLE IF EXISTS `tool_log`;
CREATE TABLE `tool_log` (
  `id`              BIGINT       NOT NULL AUTO_INCREMENT,
  `conversation_id` BIGINT       DEFAULT NULL,
  `member_id`       BIGINT       DEFAULT NULL,
  `tool_name`       VARCHAR(100) NOT NULL,
  `tool_input`      JSON         DEFAULT NULL,
  `tool_output`     JSON         DEFAULT NULL,
  `success`         TINYINT      NOT NULL DEFAULT 1,
  `duration_ms`     INT          DEFAULT NULL,
  `created_at`      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_conversation_id` (`conversation_id`),
  KEY `idx_member_id` (`member_id`),
  KEY `idx_tool_name` (`tool_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Tool 调用日志';

-- Agent 执行追踪
DROP TABLE IF EXISTS `agent_trace`;
CREATE TABLE `agent_trace` (
  `id`              BIGINT       NOT NULL AUTO_INCREMENT,
  `conversation_id` BIGINT       DEFAULT NULL,
  `member_id`       BIGINT       DEFAULT NULL,
  `agent_name`      VARCHAR(50)  NOT NULL COMMENT 'supervisor/product/order/aftersale',
  `step_name`       VARCHAR(100) DEFAULT NULL,
  `input_summary`   TEXT         DEFAULT NULL,
  `output_summary`  TEXT         DEFAULT NULL,
  `duration_ms`     INT          DEFAULT NULL,
  `created_at`      DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_conversation_id` (`conversation_id`),
  KEY `idx_agent_name` (`agent_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Agent 执行追踪';
