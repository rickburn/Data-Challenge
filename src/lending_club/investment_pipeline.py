"""
Investment Decision Pipeline
============================

This module handles investment decision making under budget constraints,
implementing various portfolio optimization strategies for loan selection.

Key Features:
- Multiple selection strategies (lowest risk, highest expected value, balanced portfolio)
- Budget constraint enforcement ($5,000 per quarter)
- Portfolio diversification controls
- Risk parameter validation
- ROI calculation and reporting
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from src.utils.logging_config import log_execution, track_data_transformation


@dataclass
class InvestmentDecision:
    """Represents a single investment decision."""
    loan_id: int
    loan_amount: float
    investment_amount: float
    default_probability: float
    expected_return: float
    risk_grade: str
    term: int
    interest_rate: float


class InvestmentDecisionMaker:
    """Makes investment decisions under budget constraints."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize investment decision maker with configuration."""
        self.config = config
        self.budget_per_quarter = config.get('budget_per_quarter', 5000.0)
        self.selection_strategy = config.get('selection_strategy', 'lowest_risk')
        self.roi_calculation_method = config.get('roi_calculation_method', 'simple')
        
        # Risk parameters
        self.max_default_probability = config.get('max_default_probability', 0.25)
        self.min_expected_return = config.get('min_expected_return', 0.05)
        
        # Portfolio diversification
        self.max_concentration_per_grade = config.get('max_concentration_per_grade', 0.30)
        self.min_loan_diversity = config.get('min_loan_diversity', 100)
        self.default_recovery_rate = config.get('default_recovery_rate', 0.30)
        
        self.logger = logging.getLogger(__name__)
    
    @log_execution
    def select_investments(self, risk_scores: np.ndarray, loan_data: pd.DataFrame,
                          budget: float) -> Dict[str, Any]:
        """Select optimal investment portfolio under budget constraints."""
        self.logger.info(f"Selecting investments with ${budget:.2f} budget using {self.selection_strategy} strategy")
        
        # Create investment candidates
        candidates = self._create_investment_candidates(risk_scores, loan_data)
        
        # Filter by risk constraints
        filtered_candidates = self._filter_by_risk_constraints(candidates)
        
        # Apply selection strategy
        selected_investments = self._apply_selection_strategy(filtered_candidates, budget)
        
        # Validate portfolio constraints
        validated_portfolio = self._validate_portfolio_constraints(selected_investments, budget)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(validated_portfolio, loan_data)
        
        self.logger.info(f"Selected {len(validated_portfolio)} investments totaling ${sum(inv.investment_amount for inv in validated_portfolio):.2f}")
        
        # Track investment decisions
        track_data_transformation(
            operation="select_investments",
            input_data={
                'total_candidates': len(candidates),
                'budget': budget,
                'strategy': self.selection_strategy
            },
            output_data={
                'selected_count': len(validated_portfolio),
                'total_investment': sum(inv.investment_amount for inv in validated_portfolio),
                'avg_risk': np.mean([inv.default_probability for inv in validated_portfolio])
            },
            metadata=portfolio_metrics
        )
        
        return {
            'selected_loans': validated_portfolio,
            'portfolio_metrics': portfolio_metrics,
            'selection_strategy': self.selection_strategy,
            'total_candidates': len(candidates),
            'filtered_candidates': len(filtered_candidates)
        }
    
    def _create_investment_candidates(self, risk_scores: np.ndarray, 
                                    loan_data: pd.DataFrame) -> List[InvestmentDecision]:
        """Create list of investment candidates from loan data and risk scores."""
        candidates = []
        
        for idx, (_, loan_row) in enumerate(loan_data.iterrows()):
            if idx >= len(risk_scores):
                break
                
            # Calculate expected return
            expected_return = self._calculate_expected_return(
                loan_row, risk_scores[idx]
            )
            
            # Create investment decision object
            loan_amount = loan_row.get('funded_amnt', loan_row.get('loan_amnt', 0))
            candidate = InvestmentDecision(
                loan_id=loan_row.get('id', idx),
                loan_amount=loan_amount,
                investment_amount=min(loan_amount, 25),  # Max $25 per loan for diversification
                default_probability=risk_scores[idx],
                expected_return=expected_return,
                risk_grade=loan_row.get('sub_grade', 'Unknown'),
                term=loan_row.get('term', 36),
                interest_rate=loan_row.get('int_rate', 0.0)
            )
            
            candidates.append(candidate)
        
        self.logger.info(f"Created {len(candidates)} investment candidates")
        return candidates
    
    def _calculate_expected_return(self, loan_row: pd.Series, default_probability: float) -> float:
        """Calculate expected return for a loan investment."""
        # Handle interest rate - convert from percentage string to decimal
        int_rate_raw = loan_row.get('int_rate', '0.0%')
        if isinstance(int_rate_raw, str) and '%' in int_rate_raw:
            interest_rate = float(int_rate_raw.strip('%')) / 100.0
        else:
            interest_rate = float(int_rate_raw)

        if self.roi_calculation_method == 'simple':
            # Simple calculation: (1 - default_prob) * interest_rate - default_prob * (1 - recovery_rate)
            expected_return = ((1 - default_probability) * interest_rate -
                             default_probability * (1 - self.default_recovery_rate))

        elif self.roi_calculation_method == 'detailed':
            # More detailed calculation considering term and payments
            term_years = loan_row.get('term', 36) / 12.0

            # Approximate annual return considering default risk
            survival_probability = 1 - default_probability
            annual_return = interest_rate * survival_probability
            recovery_on_default = default_probability * self.default_recovery_rate

            expected_return = annual_return + recovery_on_default - default_probability

        else:
            # Fallback to interest rate minus risk premium
            expected_return = interest_rate - default_probability * 2

        return expected_return
    
    def _filter_by_risk_constraints(self, candidates: List[InvestmentDecision]) -> List[InvestmentDecision]:
        """Filter candidates by risk constraints."""
        filtered = []
        failed_default_prob = 0
        failed_expected_return = 0
        failed_loan_amount = 0

        # Debug: log some sample expected returns and default probabilities
        if candidates:
            sample_candidates = candidates[:5]  # First 5 candidates
            self.logger.info(f"Sample candidate analysis:")
            for i, cand in enumerate(sample_candidates):
                self.logger.info(f"  Candidate {i}: default_prob={cand.default_probability:.4f}, "
                               f"expected_return={cand.expected_return:.4f}, loan_amount={cand.loan_amount}")

        for candidate in candidates:
            # Check maximum default probability
            if candidate.default_probability > self.max_default_probability:
                failed_default_prob += 1
                continue

            # Check minimum expected return
            if candidate.expected_return < self.min_expected_return:
                failed_expected_return += 1
                continue

            # Check for valid loan amount
            if candidate.loan_amount <= 0 or candidate.investment_amount <= 0:
                failed_loan_amount += 1
                continue

            filtered.append(candidate)

        self.logger.info(f"Risk constraint filtering summary:")
        self.logger.info(f"  Total candidates: {len(candidates)}")
        self.logger.info(f"  Failed default probability (> {self.max_default_probability}): {failed_default_prob}")
        self.logger.info(f"  Failed expected return (< {self.min_expected_return}): {failed_expected_return}")
        self.logger.info(f"  Failed loan amount: {failed_loan_amount}")
        self.logger.info(f"  Passed all constraints: {len(filtered)}")
        return filtered
    
    def _apply_selection_strategy(self, candidates: List[InvestmentDecision], 
                                budget: float) -> List[InvestmentDecision]:
        """Apply the configured selection strategy."""
        if self.selection_strategy == 'lowest_risk':
            return self._select_lowest_risk(candidates, budget)
        elif self.selection_strategy == 'highest_expected_value':
            return self._select_highest_expected_value(candidates, budget)
        elif self.selection_strategy == 'balanced_portfolio':
            return self._select_balanced_portfolio(candidates, budget)
        else:
            self.logger.warning(f"Unknown selection strategy: {self.selection_strategy}, using lowest_risk")
            return self._select_lowest_risk(candidates, budget)
    
    def _select_lowest_risk(self, candidates: List[InvestmentDecision], 
                          budget: float) -> List[InvestmentDecision]:
        """Select investments with lowest default risk."""
        # Sort by default probability (ascending)
        sorted_candidates = sorted(candidates, key=lambda x: x.default_probability)
        
        selected = []
        total_investment = 0.0
        
        for candidate in sorted_candidates:
            if total_investment + candidate.investment_amount <= budget:
                selected.append(candidate)
                total_investment += candidate.investment_amount
            else:
                # Try to fit a partial investment if possible
                remaining_budget = budget - total_investment
                if remaining_budget > 1:  # Minimum $1 investment
                    # Create partial investment
                    partial_candidate = InvestmentDecision(
                        loan_id=candidate.loan_id,
                        loan_amount=candidate.loan_amount,
                        investment_amount=remaining_budget,
                        default_probability=candidate.default_probability,
                        expected_return=candidate.expected_return,
                        risk_grade=candidate.risk_grade,
                        term=candidate.term,
                        interest_rate=candidate.interest_rate
                    )
                    selected.append(partial_candidate)
                break
        
        return selected
    
    def _select_highest_expected_value(self, candidates: List[InvestmentDecision],
                                     budget: float) -> List[InvestmentDecision]:
        """Select investments with highest expected value."""
        # Sort by expected return (descending)
        sorted_candidates = sorted(candidates, key=lambda x: x.expected_return, reverse=True)
        
        selected = []
        total_investment = 0.0
        
        for candidate in sorted_candidates:
            if total_investment + candidate.investment_amount <= budget:
                selected.append(candidate)
                total_investment += candidate.investment_amount
            else:
                # Try partial investment
                remaining_budget = budget - total_investment
                if remaining_budget > 1:
                    partial_candidate = InvestmentDecision(
                        loan_id=candidate.loan_id,
                        loan_amount=candidate.loan_amount,
                        investment_amount=remaining_budget,
                        default_probability=candidate.default_probability,
                        expected_return=candidate.expected_return,
                        risk_grade=candidate.risk_grade,
                        term=candidate.term,
                        interest_rate=candidate.interest_rate
                    )
                    selected.append(partial_candidate)
                break
        
        return selected
    
    def _select_balanced_portfolio(self, candidates: List[InvestmentDecision],
                                 budget: float) -> List[InvestmentDecision]:
        """Select balanced portfolio optimizing risk-return tradeoff."""
        # Calculate risk-adjusted score: expected_return / (default_probability + 0.01)
        # Add small epsilon to avoid division by zero
        scored_candidates = []
        for candidate in candidates:
            risk_adjusted_score = candidate.expected_return / (candidate.default_probability + 0.01)
            scored_candidates.append((candidate, risk_adjusted_score))
        
        # Sort by risk-adjusted score (descending)
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
        
        selected = []
        total_investment = 0.0
        grade_investments = {}  # Track investment by grade for diversification
        
        for candidate, score in sorted_candidates:
            # Check diversification constraint
            grade = candidate.risk_grade
            current_grade_investment = grade_investments.get(grade, 0)
            max_grade_investment = budget * self.max_concentration_per_grade
            
            available_for_grade = max_grade_investment - current_grade_investment
            investment_amount = min(candidate.investment_amount, available_for_grade)
            
            if investment_amount > 1 and total_investment + investment_amount <= budget:
                # Update investment amount if constrained by diversification
                if investment_amount < candidate.investment_amount:
                    candidate = InvestmentDecision(
                        loan_id=candidate.loan_id,
                        loan_amount=candidate.loan_amount,
                        investment_amount=investment_amount,
                        default_probability=candidate.default_probability,
                        expected_return=candidate.expected_return,
                        risk_grade=candidate.risk_grade,
                        term=candidate.term,
                        interest_rate=candidate.interest_rate
                    )
                
                selected.append(candidate)
                total_investment += investment_amount
                grade_investments[grade] = current_grade_investment + investment_amount
        
        return selected
    
    def _validate_portfolio_constraints(self, investments: List[InvestmentDecision],
                                      budget: float) -> List[InvestmentDecision]:
        """Validate and enforce portfolio constraints."""
        total_investment = sum(inv.investment_amount for inv in investments)
        
        # Check budget constraint
        if total_investment > budget:
            self.logger.warning(f"Portfolio exceeds budget: ${total_investment:.2f} > ${budget:.2f}")
            # Scale down investments proportionally
            scale_factor = budget / total_investment
            for inv in investments:
                inv.investment_amount *= scale_factor
        
        # Check minimum diversity constraint
        if len(investments) < self.min_loan_diversity:
            self.logger.info(f"Portfolio has {len(investments)} loans (minimum diversity target: {self.min_loan_diversity})")
        
        # Check grade concentration
        grade_investments = {}
        for inv in investments:
            grade = inv.risk_grade
            grade_investments[grade] = grade_investments.get(grade, 0) + inv.investment_amount
        
        max_concentration = max(grade_investments.values()) / budget if grade_investments else 0
        if max_concentration > self.max_concentration_per_grade:
            self.logger.warning(f"Grade concentration ({max_concentration:.2%}) exceeds limit ({self.max_concentration_per_grade:.2%})")
        
        return investments
    
    def _calculate_portfolio_metrics(self, investments: List[InvestmentDecision],
                                   loan_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics."""
        if not investments:
            return {
                'total_investment': 0,
                'loan_count': 0,
                'avg_risk_score': 0,
                'avg_expected_return': 0,
                'grade_distribution': {},
                'term_distribution': {},
                'risk_metrics': {
                    'portfolio_default_risk': 0.0,
                    'default_risk_std': 0.0,
                    'max_default_risk': 0.0,
                    'min_default_risk': 0.0,
                    'expected_portfolio_return': 0.0,
                    'return_std': 0.0,
                    'sharpe_ratio': 0.0,
                    'concentration_risk': 0.0
                }
            }
        
        # Basic metrics
        total_investment = sum(inv.investment_amount for inv in investments)
        loan_count = len(investments)
        avg_risk_score = np.mean([inv.default_probability for inv in investments])
        avg_expected_return = np.mean([inv.expected_return for inv in investments])
        
        # Grade distribution
        grade_dist = {}
        for inv in investments:
            grade = inv.risk_grade
            grade_dist[grade] = grade_dist.get(grade, 0) + inv.investment_amount
        
        # Convert to percentages
        grade_distribution = {grade: amount/total_investment for grade, amount in grade_dist.items()}
        
        # Term distribution
        term_dist = {}
        for inv in investments:
            term = inv.term
            term_dist[term] = term_dist.get(term, 0) + inv.investment_amount
        
        term_distribution = {term: amount/total_investment for term, amount in term_dist.items()}
        
        # Risk metrics
        risk_scores = [inv.default_probability for inv in investments] if investments else []
        expected_returns = [inv.expected_return for inv in investments] if investments else []

        if investments:
            risk_metrics = {
                'portfolio_default_risk': np.mean(risk_scores),
                'default_risk_std': np.std(risk_scores),
                'max_default_risk': np.max(risk_scores),
                'min_default_risk': np.min(risk_scores),
                'expected_portfolio_return': np.mean(expected_returns),
                'return_std': np.std(expected_returns),
                'sharpe_ratio': np.mean(expected_returns) / (np.std(expected_returns) + 1e-8),
                'concentration_risk': max(grade_distribution.values()) if grade_distribution else 0
            }
        else:
            risk_metrics = {
                'portfolio_default_risk': 0.0,
                'default_risk_std': 0.0,
                'max_default_risk': 0.0,
                'min_default_risk': 0.0,
                'expected_portfolio_return': 0.0,
                'return_std': 0.0,
                'sharpe_ratio': 0.0,
                'concentration_risk': 0.0
            }
        
        return {
            'total_investment': total_investment,
            'loan_count': loan_count,
            'avg_risk_score': avg_risk_score,
            'avg_expected_return': avg_expected_return,
            'grade_distribution': grade_distribution,
            'term_distribution': term_distribution,
            'risk_metrics': risk_metrics
        }
    
    def generate_decision_summary(self, investment_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of investment decisions."""
        selected_loans = investment_decisions['selected_loans']
        portfolio_metrics = investment_decisions['portfolio_metrics']
        
        summary = {
            'selection_strategy': self.selection_strategy,
            'budget_allocated': self.budget_per_quarter,
            'budget_used': portfolio_metrics['total_investment'],
            'total_investment': portfolio_metrics['total_investment'],  # Add missing key
            'budget_utilization': portfolio_metrics['total_investment'] / self.budget_per_quarter,
            'loans_selected': len(selected_loans),
            'avg_investment_per_loan': portfolio_metrics['total_investment'] / len(selected_loans) if selected_loans else 0,
            'portfolio_risk_profile': {
                'avg_default_probability': portfolio_metrics['avg_risk_score'],
                'risk_range': {
                    'min': portfolio_metrics['risk_metrics']['min_default_risk'],
                    'max': portfolio_metrics['risk_metrics']['max_default_risk'],
                    'std': portfolio_metrics['risk_metrics']['default_risk_std']
                }
            },
            'expected_performance': {
                'avg_expected_return': portfolio_metrics['avg_expected_return'],
                'portfolio_sharpe_ratio': portfolio_metrics['risk_metrics']['sharpe_ratio'],
                'estimated_annual_return': portfolio_metrics['avg_expected_return'] * portfolio_metrics['total_investment']
            },
            'diversification_metrics': {
                'grade_concentration': portfolio_metrics['risk_metrics']['concentration_risk'],
                'grade_distribution': portfolio_metrics['grade_distribution'],
                'term_distribution': portfolio_metrics['term_distribution']
            }
        }
        
        # Add compliance checks
        summary['compliance_status'] = {
            'budget_constraint': summary['budget_used'] <= self.budget_per_quarter,
            'max_default_risk': portfolio_metrics['risk_metrics']['max_default_risk'] <= self.max_default_probability,
            'min_expected_return': portfolio_metrics['avg_expected_return'] >= self.min_expected_return,
            'grade_concentration': portfolio_metrics['risk_metrics']['concentration_risk'] <= self.max_concentration_per_grade
        }
        
        return summary
